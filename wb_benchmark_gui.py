import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import numpy as np
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading

warnings.filterwarnings('ignore')


class DeepWBModel(nn.Module):
    """Упрощенная реализация DeepWB модели"""
    
    def __init__(self):
        super(DeepWBModel, self).__init__()
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Предсказание коэффициентов коррекции
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),  # RGB коэффициенты
            nn.Softplus()  # Обеспечиваем положительные значения
        )
    
    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        gains = self.fc(features)
        
        # Применяем коэффициенты к изображению
        gains = gains.view(-1, 3, 1, 1)
        corrected = x * gains
        
        return torch.clamp(corrected, 0, 1)


class FFCCModel(nn.Module):
    """Упрощенная реализация FFCC модели"""
    
    def __init__(self):
        super(FFCCModel, self).__init__()
        # CNN для извлечения признаков
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Гистограммные признаки
        self.histogram_net = nn.Sequential(
            nn.Linear(128 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True)
        )
        
        # Предсказание освещения
        self.illuminant_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Извлекаем признаки
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        # Обрабатываем гистограммные признаки
        hist_features = self.histogram_net(features)
        
        # Предсказываем освещение
        illuminant = self.illuminant_net(hist_features)
        
        # Коррекция изображения
        illuminant = illuminant.view(-1, 3, 1, 1)
        # Нормализуем по зеленому каналу
        normalized_illuminant = illuminant / (illuminant[:, 1:2] + 1e-8)
        corrected = x / (normalized_illuminant + 1e-8)
        
        return torch.clamp(corrected, 0, 1)


class WhiteBalanceAlgorithms:
    """Классические алгоритмы баланса белого на PyTorch"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def gray_world(self, image: torch.Tensor) -> torch.Tensor:
        """Gray World алгоритм"""
        mean_rgb = torch.mean(image, dim=(2, 3), keepdim=True)
        gray_mean = torch.mean(mean_rgb, dim=1, keepdim=True)
        gain = gray_mean / (mean_rgb + 1e-8)
        corrected = image * gain
        return torch.clamp(corrected, 0, 1)
    
    def white_patch(self, image: torch.Tensor, percentile: float = 99.0) -> torch.Tensor:
        """White Patch алгоритм"""
        batch_size, channels, height, width = image.shape
        image_flat = image.view(batch_size, channels, -1)
        
        if percentile < 100:
            max_vals = torch.quantile(image_flat, percentile/100.0, dim=2, keepdim=True)
        else:
            max_vals = torch.max(image_flat, dim=2, keepdim=True)[0]
        
        max_vals = max_vals.unsqueeze(3)
        gain = torch.max(max_vals) / (max_vals + 1e-8)
        
        corrected = image * gain
        return torch.clamp(corrected, 0, 1)
    
    def shades_of_gray(self, image: torch.Tensor, p: float = 6.0) -> torch.Tensor:
        """Shades of Gray алгоритм"""
        powered = torch.pow(image + 1e-8, p)
        lp_norm = torch.pow(torch.mean(powered, dim=(2, 3), keepdim=True), 1.0/p)
        gray_mean = torch.mean(lp_norm, dim=1, keepdim=True)
        gain = gray_mean / (lp_norm + 1e-8)
        corrected = image * gain
        return torch.clamp(corrected, 0, 1)
    
    def color_by_correlation(self, image: torch.Tensor, reference_color: torch.Tensor = None) -> torch.Tensor:
        """Color by Correlation алгоритм"""
        if reference_color is None:
            reference_color = torch.mean(image, dim=(2, 3), keepdim=True)
        
        r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
        r_mean = torch.mean(r, dim=(2, 3), keepdim=True)
        g_mean = torch.mean(g, dim=(2, 3), keepdim=True)
        b_mean = torch.mean(b, dim=(2, 3), keepdim=True)
        
        r_gain = g_mean / (r_mean + 1e-8)
        b_gain = g_mean / (b_mean + 1e-8)
        
        corrected = torch.cat([r * r_gain, g, b * b_gain], dim=1)
        return torch.clamp(corrected, 0, 1)
    
    def histogram_based_wb(self, image: torch.Tensor, bins: int = 256) -> torch.Tensor:
        """Гистограммный метод баланса белого"""
        batch_size, channels, height, width = image.shape
        corrected_channels = []
        
        for c in range(channels):
            channel = image[:, c:c+1]
            hist = torch.histc(channel.view(-1), bins=bins, min=0, max=1)
            cumsum = torch.cumsum(hist, dim=0)
            total = cumsum[-1]
            
            low_idx = torch.where(cumsum >= 0.01 * total)[0][0]
            high_idx = torch.where(cumsum >= 0.99 * total)[0][0]
            
            low_val = low_idx.float() / bins
            high_val = high_idx.float() / bins
            
            channel_corrected = (channel - low_val) / (high_val - low_val + 1e-8)
            corrected_channels.append(torch.clamp(channel_corrected, 0, 1))
        
        return torch.cat(corrected_channels, dim=1)


class QualityMetrics:
    """Метрики качества для оценки алгоритмов баланса белого"""
    
    @staticmethod
    def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
        """Вычисление Structural Similarity Index (SSIM)"""
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2
        
        batch_size, channels, height, width = img1.shape
        total_ssim = 0.0
        
        x_data, y_data = torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing='xy'
        )
        gauss_kernel = torch.exp(
            -((x_data - window_size//2)**2 + (y_data - window_size//2)**2) / 
            (2 * sigma**2)
        )
        gauss_kernel = gauss_kernel / gauss_kernel.sum()
        window = gauss_kernel.view(1, 1, window_size, window_size).to(img1.device)
        
        for c in range(channels):
            channel1 = img1[:, c:c+1]
            channel2 = img2[:, c:c+1]
            
            mu1 = F.conv2d(channel1, window, padding=window_size//2)
            mu2 = F.conv2d(channel2, window, padding=window_size//2)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.conv2d(channel1 * channel1, window, padding=window_size//2) - mu1_sq
            sigma2_sq = F.conv2d(channel2 * channel2, window, padding=window_size//2) - mu2_sq
            sigma12 = F.conv2d(channel1 * channel2, window, padding=window_size//2) - mu1_mu2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            total_ssim += torch.mean(ssim_map).item()
        
        return total_ssim / channels
    
    @staticmethod
    def delta_e_cie76(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Цветовая ошибка Delta E CIE76"""
        lab1 = QualityMetrics.rgb_to_lab(img1)
        lab2 = QualityMetrics.rgb_to_lab(img2)
        delta_e = torch.sqrt(torch.sum((lab1 - lab2) ** 2, dim=1))
        return torch.mean(delta_e).item()
    
    @staticmethod
    def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
        """Упрощенная конвертация RGB в Lab"""
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        l = 0.299 * r + 0.587 * g + 0.114 * b
        a = 0.5 * (r - g)
        b_lab = 0.5 * (g - b)
        return torch.stack([l, a, b_lab], dim=1)
    
    @staticmethod
    def angular_error(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Угловая ошибка между изображениями"""
        img1_norm = F.normalize(img1.view(img1.size(0), -1), dim=1)
        img2_norm = F.normalize(img2.view(img2.size(0), -1), dim=1)
        cos_angle = torch.sum(img1_norm * img2_norm, dim=1)
        cos_angle = torch.clamp(cos_angle, -1, 1)
        angle = torch.acos(cos_angle) * 180 / torch.pi
        return torch.mean(angle).item()
    
    @staticmethod
    def mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Mean Squared Error"""
        return F.mse_loss(img1, img2).item()
    
    @staticmethod
    def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Peak Signal-to-Noise Ratio"""
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


class WhiteBalanceBenchmarkGUI:
    """GUI для бенчмарка алгоритмов баланса белого"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.wb_algorithms = WhiteBalanceAlgorithms(self.device)
        
        # Инициализация ML моделей
        self.deepwb_model = DeepWBModel().to(self.device)
        self.ffcc_model = FFCCModel().to(self.device)
        
        # Переводим модели в режим оценки
        self.deepwb_model.eval()
        self.ffcc_model.eval()
        
        # Классические алгоритмы
        self.classical_algorithms = {
            'GW': self.wb_algorithms.gray_world,
            'WP': self.wb_algorithms.white_patch,
            'SoG': self.wb_algorithms.shades_of_gray,
            'CbC': self.wb_algorithms.color_by_correlation,
            'HB': self.wb_algorithms.histogram_based_wb,
        }
        
        # ML алгоритмы
        self.ml_algorithms = {
            'DeepWB': self.deepwb_inference,
            'FFCC': self.ffcc_inference,
        }
        
        # Результаты бенчмарка
        self.classical_results = None
        self.ml_results = None
        self.comparison_results = None
        
        # Создание GUI
        self.setup_gui()
        
    def deepwb_inference(self, image: torch.Tensor) -> torch.Tensor:
        """Инференс для DeepWB модели"""
        with torch.no_grad():
            return self.deepwb_model(image)
    
    def ffcc_inference(self, image: torch.Tensor) -> torch.Tensor:
        """Инференс для FFCC модели"""
        with torch.no_grad():
            return self.ffcc_model(image)
    
    def setup_gui(self):
        """Создание GUI интерфейса"""
        self.root = tk.Tk()
        self.root.title("White Balance Benchmark Suite")
        self.root.geometry("1400x900")
        
        # Создание notebook для вкладок
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Вкладки
        self.setup_control_tab()
        self.setup_classical_results_tab()
        self.setup_ml_results_tab()
        self.setup_comparison_tab()
        
    def setup_control_tab(self):
        """Вкладка управления"""
        control_frame = ttk.Frame(self.notebook)
        self.notebook.add(control_frame, text="Управление")
        
        # Область загрузки изображений
        load_frame = ttk.LabelFrame(control_frame, text="Загрузка изображений", padding=10)
        load_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(load_frame, text="Выбрать изображения", 
                  command=self.load_images).pack(side=tk.LEFT, padx=5)
        
        self.image_count_label = ttk.Label(load_frame, text="Изображений загружено: 0")
        self.image_count_label.pack(side=tk.LEFT, padx=20)
        
        # Область настроек
        settings_frame = ttk.LabelFrame(control_frame, text="Настройки бенчмарка", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Количество прогонов
        ttk.Label(settings_frame, text="Количество прогонов:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.num_runs_var = tk.IntVar(value=5)
        ttk.Spinbox(settings_frame, from_=1, to=20, width=10, 
                   textvariable=self.num_runs_var).grid(row=0, column=1, padx=5)
        
        # Размер изображения
        ttk.Label(settings_frame, text="Размер изображения:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.image_size_var = tk.StringVar(value="512x512")
        size_combo = ttk.Combobox(settings_frame, textvariable=self.image_size_var,
                                 values=["256x256", "512x512", "768x768", "1024x1024"])
        size_combo.grid(row=1, column=1, padx=5)
        
        # Кнопки запуска
        buttons_frame = ttk.LabelFrame(control_frame, text="Запуск бенчмарка", padding=10)
        buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(buttons_frame, text="Запустить классические алгоритмы",
                  command=self.run_classical_benchmark).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(buttons_frame, text="Запустить ML алгоритмы",
                  command=self.run_ml_benchmark).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(buttons_frame, text="Создать сравнение",
                  command=self.create_comparison).pack(side=tk.LEFT, padx=5)
        
        # Прогресс бар
        progress_frame = ttk.LabelFrame(control_frame, text="Прогресс", padding=10)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                          maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(progress_frame, text="Готов к работе")
        self.status_label.pack(pady=5)
        
        # Инициализация данных
        self.image_paths = []
        
    def setup_classical_results_tab(self):
        """Вкладка результатов классических алгоритмов"""
        self.classical_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.classical_frame, text="Классические алгоритмы")
        
        # Таблица результатов
        columns = ('Algorithm', 'Time (ms)', 'Delta E', 'Angular Error', 'MSE', 'PSNR', 'SSIM')
        self.classical_tree = ttk.Treeview(self.classical_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.classical_tree.heading(col, text=col)
            self.classical_tree.column(col, width=100)
        
        self.classical_tree.pack(fill=tk.X, padx=10, pady=5)
        
        # График
        self.classical_fig = Figure(figsize=(12, 8))
        self.classical_canvas = FigureCanvasTkAgg(self.classical_fig, self.classical_frame)
        self.classical_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Кнопки сохранения
        save_frame = ttk.Frame(self.classical_frame)
        save_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(save_frame, text="Сохранить результаты CSV",
                  command=lambda: self.save_results('classical')).pack(side=tk.LEFT, padx=5)
        ttk.Button(save_frame, text="Сохранить график",
                  command=lambda: self.save_plot('classical')).pack(side=tk.LEFT, padx=5)
        
    def setup_ml_results_tab(self):
        """Вкладка результатов ML алгоритмов"""
        self.ml_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ml_frame, text="ML алгоритмы")
        
        # Таблица результатов
        columns = ('Algorithm', 'Time (ms)', 'Delta E', 'Angular Error', 'MSE', 'PSNR', 'SSIM')
        self.ml_tree = ttk.Treeview(self.ml_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.ml_tree.heading(col, text=col)
            self.ml_tree.column(col, width=100)
        
        self.ml_tree.pack(fill=tk.X, padx=10, pady=5)
        
        # График
        self.ml_fig = Figure(figsize=(12, 8))
        self.ml_canvas = FigureCanvasTkAgg(self.ml_fig, self.ml_frame)
        self.ml_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Кнопки сохранения
        save_frame = ttk.Frame(self.ml_frame)
        save_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(save_frame, text="Сохранить результаты CSV",
                  command=lambda: self.save_results('ml')).pack(side=tk.LEFT, padx=5)
        ttk.Button(save_frame, text="Сохранить график",
                  command=lambda: self.save_plot('ml')).pack(side=tk.LEFT, padx=5)
        
    def setup_comparison_tab(self):
        """Вкладка сравнения"""
        self.comparison_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.comparison_frame, text="Сравнение")
        
        # Текстовая область для сравнительной таблицы
        text_frame = ttk.LabelFrame(self.comparison_frame, text="Сравнительная таблица", padding=10)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.comparison_text = tk.Text(text_frame, wrap=tk.NONE, font=('Courier', 10))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.comparison_text.yview)
        h_scrollbar = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=self.comparison_text.xview)
        self.comparison_text.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.comparison_text.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        
        # Кнопки
        button_frame = ttk.Frame(self.comparison_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Сохранить сравнение",
                  command=lambda: self.save_results('comparison')).pack(side=tk.LEFT, padx=5)
        
    def load_images(self):
        """Загрузка изображений"""
        file_paths = filedialog.askopenfilenames(
            title="Выберите изображения",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_paths:
            self.image_paths = list(file_paths)
            self.image_count_label.config(text=f"Изображений загружено: {len(self.image_paths)}")
        
    def load_image(self, image_path: str) -> torch.Tensor:
        """Загрузка и предобработка изображения"""
        size_str = self.image_size_var.get()
        size = tuple(map(int, size_str.split('x')))
        
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])
        
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def simulate_color_cast(self, image: torch.Tensor, temperature: float = 2800) -> torch.Tensor:
        """Симуляция цветового сдвига"""
        if temperature < 5500:  # Warm cast
            gain = torch.tensor([1.2, 1.0, 0.8]).to(self.device)
        else:  # Cool cast
            gain = torch.tensor([0.8, 1.0, 1.2]).to(self.device)
        
        gain = gain.view(1, 3, 1, 1)
        return torch.clamp(image * gain, 0, 1)
    
    def benchmark_algorithm(self, algorithm_name: str, algorithm_func,
                          images: List[torch.Tensor], ground_truth: List[torch.Tensor]) -> Dict:
        """Бенчмарк одного алгоритма"""
        num_runs = self.num_runs_var.get()
        
        results = {
            'algorithm': algorithm_name,
            'avg_time': 0,
            'delta_e': 0,
            'angular_error': 0,
            'mse': 0,
            'psnr': 0,
            'ssim': 0,
            'num_images': len(images)
        }
        
        total_time = 0
        total_delta_e = 0
        total_angular_error = 0
        total_mse = 0
        total_psnr = 0
        total_ssim = 0
        
        for img, gt in zip(images, ground_truth):
            # Измеряем время
            times = []
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                corrected = algorithm_func(img)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            total_time += avg_time
            
            # Вычисляем метрики качества
            delta_e = QualityMetrics.delta_e_cie76(corrected, gt)
            angular_error = QualityMetrics.angular_error(corrected, gt)
            mse = QualityMetrics.mse(corrected, gt)
            psnr = QualityMetrics.psnr(corrected, gt)
            ssim_val = QualityMetrics.ssim(corrected, gt)
            
            total_delta_e += delta_e
            total_angular_error += angular_error
            total_mse += mse
            total_psnr += psnr
            total_ssim += ssim_val
        
        # Усредняем по всем изображениям
        num_images = len(images)
        results['avg_time'] = total_time / num_images
        results['delta_e'] = total_delta_e / num_images
        results['angular_error'] = total_angular_error / num_images
        results['mse'] = total_mse / num_images
        results['psnr'] = total_psnr / num_images
        results['ssim'] = total_ssim / num_images
        
        return results
    
    def run_classical_benchmark(self):
        """Запуск бенчмарка классических алгоритмов"""
        if not self.image_paths:
            messagebox.showerror("Ошибка", "Пожалуйста, загрузите изображения")
            return
        
        def benchmark_thread():
            try:
                self.status_label.config(text="Подготовка изображений...")
                self.progress_var.set(0)
                
                # Загружаем изображения
                images = []
                ground_truth = []
                
                for i, path in enumerate(self.image_paths):
                    original = self.load_image(path)
                    cast_image = self.simulate_color_cast(original)
                    
                    images.append(cast_image)
                    ground_truth.append(original)
                    
                    progress = (i + 1) / len(self.image_paths) * 20  # 20% на загрузку
                    self.progress_var.set(progress)
                    self.root.update()
                
                # Запускаем бенчмарк
                results = []
                total_algorithms = len(self.classical_algorithms)
                
                for i, (algo_name, algo_func) in enumerate(self.classical_algorithms.items()):
                    self.status_label.config(text=f"Тестирование {algo_name}...")
                    
                    result = self.benchmark_algorithm(algo_name, algo_func, images, ground_truth)
                    results.append(result)
                    
                    progress = 20 + (i + 1) / total_algorithms * 80  # 80% на бенчмарк
                    self.progress_var.set(progress)
                    self.root.update()
                
                self.classical_results = pd.DataFrame(results)
                self.update_classical_results()
                
                self.status_label.config(text="Бенчмарк классических алгоритмов завершен")
                self.progress_var.set(100)
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при выполнении бенчмарка: {str(e)}")
                self.status_label.config(text="Ошибка выполнения")
        
        threading.Thread(target=benchmark_thread, daemon=True).start()
    
    def run_ml_benchmark(self):
        """Запуск бенчмарка ML алгоритмов"""
        if not self.image_paths:
            messagebox.showerror("Ошибка", "Пожалуйста, загрузите изображения")
            return
        
        def benchmark_thread():
            try:
                self.status_label.config(text="Подготовка изображений для ML...")
                self.progress_var.set(0)
                
                # Загружаем изображения
                images = []
                ground_truth = []
                
                for i, path in enumerate(self.image_paths):
                    original = self.load_image(path)
                    cast_image = self.simulate_color_cast(original)
                    
                    images.append(cast_image)
                    ground_truth.append(original)
                    
                    progress = (i + 1) / len(self.image_paths) * 20
                    self.progress_var.set(progress)
                    self.root.update()
                
                # Запускаем бенчмарк ML алгоритмов
                results = []
                total_algorithms = len(self.ml_algorithms)
                
                for i, (algo_name, algo_func) in enumerate(self.ml_algorithms.items()):
                    self.status_label.config(text=f"Тестирование {algo_name}...")
                    
                    result = self.benchmark_algorithm(algo_name, algo_func, images, ground_truth)
                    results.append(result)
                    
                    progress = 20 + (i + 1) / total_algorithms * 80
                    self.progress_var.set(progress)
                    self.root.update()
                
                self.ml_results = pd.DataFrame(results)
                self.update_ml_results()
                
                self.status_label.config(text="Бенчмарк ML алгоритмов завершен")
                self.progress_var.set(100)
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при выполнении ML бенчмарка: {str(e)}")
                self.status_label.config(text="Ошибка выполнения")
        
        threading.Thread(target=benchmark_thread, daemon=True).start()
    
    def update_classical_results(self):
        """Обновление результатов классических алгоритмов"""
        # Очищаем таблицу
        for item in self.classical_tree.get_children():
            self.classical_tree.delete(item)
        
        # Заполняем таблицу
        for _, row in self.classical_results.iterrows():
            self.classical_tree.insert('', tk.END, values=(
                row['algorithm'],
                f"{row['avg_time']*1000:.2f}",
                f"{row['delta_e']:.3f}",
                f"{row['angular_error']:.2f}",
                f"{row['mse']:.4f}",
                f"{row['psnr']:.2f}",
                f"{row['ssim']:.3f}"
            ))
        
        # Обновляем график
        self.plot_results(self.classical_results, self.classical_fig)
        self.classical_canvas.draw()
    
    def update_ml_results(self):
        """Обновление результатов ML алгоритмов"""
        # Очищаем таблицу
        for item in self.ml_tree.get_children():
            self.ml_tree.delete(item)
        
        # Заполняем таблицу
        for _, row in self.ml_results.iterrows():
            self.ml_tree.insert('', tk.END, values=(
                row['algorithm'],
                f"{row['avg_time']*1000:.2f}",
                f"{row['delta_e']:.3f}",
                f"{row['angular_error']:.2f}",
                f"{row['mse']:.4f}",
                f"{row['psnr']:.2f}",
                f"{row['ssim']:.3f}"
            ))
        
        # Обновляем график
        self.plot_results(self.ml_results, self.ml_fig)
        self.ml_canvas.draw()
    
    def plot_results(self, df: pd.DataFrame, fig: Figure):
        """Создание графиков результатов"""
        fig.clear()
    
        # Устанавливаем размер фигуры
        fig.set_size_inches(15, 10)
    
        # Создаем подграфики БЕЗ параметра figsize
        axes = fig.subplots(2, 3)
    
        # График времени выполнения
        axes[0, 0].bar(df['algorithm'], df['avg_time'] * 1000)
        axes[0, 0].set_title('Время выполнения (мс)')
        axes[0, 0].set_ylabel('Время (мс)')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
        # График Delta E
        axes[0, 1].bar(df['algorithm'], df['delta_e'])
        axes[0, 1].set_title('Цветовая ошибка (Delta E)')
        axes[0, 1].set_ylabel('Delta E')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
        # График Angular Error
        axes[0, 2].bar(df['algorithm'], df['angular_error'])
        axes[0, 2].set_title('Угловая ошибка (градусы)')
        axes[0, 2].set_ylabel('Угол (градусы)')
        axes[0, 2].tick_params(axis='x', rotation=45)
    
        # График MSE
        axes[1, 0].bar(df['algorithm'], df['mse'])
        axes[1, 0].set_title('Среднеквадратичная ошибка')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
        # График PSNR
        axes[1, 1].bar(df['algorithm'], df['psnr'])
        axes[1, 1].set_title('PSNR (дБ)')
        axes[1, 1].set_ylabel('PSNR (дБ)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
        # График SSIM
        axes[1, 2].bar(df['algorithm'], df['ssim'])
        axes[1, 2].set_title('Структурное сходство (SSIM)')
        axes[1, 2].set_ylabel('SSIM')
        axes[1, 2].tick_params(axis='x', rotation=45)
    
        fig.tight_layout()
    
    def create_comparison(self):
        """Создание сравнительной таблицы"""
        if self.classical_results is None or self.ml_results is None:
            messagebox.showerror("Ошибка", 
                               "Пожалуйста, сначала запустите бенчмарки классических и ML алгоритмов")
            return
        
        # Находим лучшие классические алгоритмы
        classical_sorted = self.classical_results.sort_values('delta_e')
        best_classical = classical_sorted.head(2)
        
        # Объединяем результаты
        comparison_df = pd.concat([best_classical, self.ml_results], ignore_index=True)
        
        # Создаем сравнительную таблицу
        comparison_text = self.create_comparison_table(comparison_df)
        
        # Обновляем текстовое поле
        self.comparison_text.delete(1.0, tk.END)
        self.comparison_text.insert(1.0, comparison_text)
        
        # Сохраняем результаты для экспорта
        self.comparison_results = comparison_df
        
        # Выводим рекомендации
        self.add_recommendations(comparison_df)
    
    def create_comparison_table(self, df: pd.DataFrame) -> str:
        """Создание текстовой сравнительной таблицы"""
        # Сортируем по качеству (меньше Delta E = лучше)
        df_sorted = df.sort_values('delta_e')
        
        table = "=" * 120 + "\n"
        table += "СРАВНИТЕЛЬНАЯ ТАБЛИЦА АЛГОРИТМОВ БАЛАНСА БЕЛОГО\n"
        table += "=" * 120 + "\n\n"
        
        table += f"{'Алгоритм':<20} | {'Время (мс)':<12} | {'Delta E':<10} | {'Угловая ошибка':<15} | {'MSE':<12} | {'PSNR':<8} | {'SSIM':<8} | {'Ранг':<6}\n"
        table += "-" * 120 + "\n"
        
        for idx, (_, row) in enumerate(df_sorted.iterrows()):
            table += f"{row['algorithm']:<20} | {row['avg_time']*1000:>10.2f}  | {row['delta_e']:>8.3f}  | {row['angular_error']:>13.2f}°  | {row['mse']:>10.4f}  | {row['psnr']:>6.2f}  | {row['ssim']:>6.3f}  | {idx+1:>4}\n"
        
        return table
    
    def add_recommendations(self, df: pd.DataFrame):
        """Добавление рекомендаций в сравнительную таблицу"""
        best_quality = df.loc[df['delta_e'].idxmin()]
        fastest = df.loc[df['avg_time'].idxmin()]
        
        # Вычисляем балансный показатель
        df_temp = df.copy()
        df_temp['balance_score'] = 1 / (df_temp['delta_e'] * df_temp['avg_time'])
        best_balance = df_temp.loc[df_temp['balance_score'].idxmax()]
        
        recommendations = f"""

{"-" * 120}
РЕКОМЕНДАЦИИ:
{"-" * 120}

🏆 ЛУЧШЕЕ КАЧЕСТВО:
   Алгоритм: {best_quality['algorithm']}
   Delta E: {best_quality['delta_e']:.3f}
   Время: {best_quality['avg_time']*1000:.2f} мс
   SSIM: {best_quality['ssim']:.3f}

⚡ САМЫЙ БЫСТРЫЙ:
   Алгоритм: {fastest['algorithm']}
   Время: {fastest['avg_time']*1000:.2f} мс
   Delta E: {fastest['delta_e']:.3f}
   SSIM: {fastest['ssim']:.3f}

⚖️ ЛУЧШИЙ БАЛАНС (качество/скорость):
   Алгоритм: {best_balance['algorithm']}
   Балансный показатель: {best_balance['balance_score']:.2f}
   Delta E: {best_balance['delta_e']:.3f}
   Время: {best_balance['avg_time']*1000:.2f} мс

{"-" * 120}
АНАЛИЗ РЕЗУЛЬТАТОВ:
{"-" * 120}
"""
        
        # Анализ по категориям
        classical_algos = df[df['algorithm'].isin(['Gray World', 'White Patch', 'Shades of Gray', 
                                                  'Color by Correlation', 'Histogram Based'])]
        ml_algos = df[df['algorithm'].isin(['DeepWB', 'FFCC'])]
        
        if not classical_algos.empty and not ml_algos.empty:
            classical_avg_delta_e = classical_algos['delta_e'].mean()
            ml_avg_delta_e = ml_algos['delta_e'].mean()
            
            classical_avg_time = classical_algos['avg_time'].mean() * 1000
            ml_avg_time = ml_algos['avg_time'].mean() * 1000
            
            recommendations += f"""
📊 Классические алгоритмы:
   Средняя Delta E: {classical_avg_delta_e:.3f}
   Среднее время: {classical_avg_time:.2f} мс
   Лучший: {classical_algos.loc[classical_algos['delta_e'].idxmin(), 'algorithm']}

🤖 ML алгоритмы:
   Средняя Delta E: {ml_avg_delta_e:.3f}
   Среднее время: {ml_avg_time:.2f} мс
   Лучший: {ml_algos.loc[ml_algos['delta_e'].idxmin(), 'algorithm']}

💡 ВЫВОД:
   {'ML алгоритмы показывают лучшее качество' if ml_avg_delta_e < classical_avg_delta_e else 'Классические алгоритмы показывают лучшее качество'}
   {'ML алгоритмы работают быстрее' if ml_avg_time < classical_avg_time else 'Классические алгоритмы работают быстрее'}
"""
        
        self.comparison_text.insert(tk.END, recommendations)
    
    def save_results(self, result_type: str):
        """Сохранение результатов"""
        if result_type == 'classical' and self.classical_results is not None:
            df = self.classical_results
            title = "классических алгоритмов"
        elif result_type == 'ml' and self.ml_results is not None:
            df = self.ml_results
            title = "ML алгоритмов"
        elif result_type == 'comparison' and self.comparison_results is not None:
            # Сохраняем текст сравнения
            filename = filedialog.asksaveasfilename(
                title="Сохранить сравнительную таблицу",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                defaultextension=".txt"
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.comparison_text.get(1.0, tk.END))
                messagebox.showinfo("Успех", "Сравнительная таблица сохранена!")
            return
        else:
            messagebox.showerror("Ошибка", "Нет данных для сохранения")
            return
        
        filename = filedialog.asksaveasfilename(
            title=f"Сохранить результаты {title}",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")],
            defaultextension=".csv"
        )
        
        if filename:
            if filename.endswith('.xlsx'):
                df.to_excel(filename, index=False)
            else:
                df.to_csv(filename, index=False, encoding='utf-8')
            messagebox.showinfo("Успех", f"Результаты {title} сохранены!")
    
    def save_plot(self, plot_type: str):
        """Сохранение графиков"""
        filename = filedialog.asksaveasfilename(
            title=f"Сохранить график {plot_type}",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")],
            defaultextension=".png"
        )
        
        if filename:
            if plot_type == 'classical' and self.classical_results is not None:
                self.classical_fig.savefig(filename, dpi=300, bbox_inches='tight')
            elif plot_type == 'ml' and self.ml_results is not None:
                self.ml_fig.savefig(filename, dpi=300, bbox_inches='tight')
            
            messagebox.showinfo("Успех", "График сохранен!")
    
    def run(self):
        """Запуск GUI"""
        self.root.mainloop()


def main():
    """Основная функция"""
    try:
        app = WhiteBalanceBenchmarkGUI()
        app.run()
    except Exception as e:
        print(f"Ошибка при запуске приложения: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()