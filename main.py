import tkinter as tk
from tkinter import filedialog, Scale, HORIZONTAL, IntVar, DoubleVar
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np

class ImageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Filter Application")
        self.root.geometry("1000x700")

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TButton", font=("Helvetica", 10), padding=6, relief="flat", background="#4CAF50", foreground="#FFFFFF")
        self.style.map("TButton", background=[("active", "#45A049")])

        self.style.configure("TLabel", font=("Helvetica", 12), background="#F0F0F0")
        self.style.configure("TFrame", background="#F0F0F0")
        self.style.configure("TCombobox", font=("Helvetica", 11), padding=6)

        self.original_image = None
        self.filtered_image = None

        # Initialize Hough Circle parameters
        self.dp_var = DoubleVar(value=1.2)
        self.min_dist_var = IntVar(value=20)
        self.param1_var = IntVar(value=50)
        self.param2_var = IntVar(value=30)
        
        self.filter_map = {
            "Blur": self.apply_blur,
            "Gaussian Blur": self.apply_gaussian,
            "Median Blur": self.apply_median,
            "Sobel": self.apply_sobel_edge_detector,
            "Prewitt": self.apply_prewitt_edge_detector,
            "Erosion": self.apply_erosion,
            "Hough Circles": self.apply_hough_circles,
            "Dilation": self.apply_dilation,
            "High pass filter": self.apply_hpf
        }

        self.init_ui()

    def init_ui(self):
        # Main frames
        frame_image = ttk.Frame(self.root, padding=10)
        frame_image.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        frame_control = ttk.Frame(self.root, padding=10)
        frame_control.pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas for original and filtered images
        self.canvas_original = tk.Canvas(frame_image, width=600, height=600, bg='white', highlightthickness=1, highlightbackground="#4CAF50")
        self.canvas_original.grid(row=0, column=0, padx=10, pady=10)

        self.canvas_filtered = tk.Canvas(frame_image, width=600, height=600, bg='white', highlightthickness=1, highlightbackground="#4CAF50")
        self.canvas_filtered.grid(row=0, column=1, padx=10, pady=10)

        # Control panel
        ttk.Button(frame_control, text="Upload Image", command=self.upload_image).grid(row=0, column=0, padx=10, pady=10)

        ttk.Label(frame_control, text="Select Filter:").grid(row=0, column=1, padx=10)
        self.filter_combo = ttk.Combobox(frame_control, values=list(self.filter_map.keys()), state="readonly")
        self.filter_combo.grid(row=0, column=2, padx=10, pady=10)
        self.filter_combo.current(0)
        self.filter_combo.bind("<<ComboboxSelected>>", self.toggle_hough_controls)

        ttk.Label(frame_control, text="Kernel Size:").grid(row=0, column=3, padx=10)
        self.kernel_slider = Scale(frame_control, from_=1, to=31, orient=HORIZONTAL, resolution=2, command=self.update_filter, bg="#F0F0F0", highlightthickness=0)
        self.kernel_slider.set(3)
        self.kernel_slider.grid(row=0, column=4, padx=10, pady=10)

        # Hough Circle Controls
        self.frame_hough = ttk.Frame(frame_control)
        self.frame_hough.grid(row=1, column=0, columnspan=5, pady=10)

        ttk.Label(self.frame_hough, text="dp:").grid(row=0, column=0, padx=5)
        self.dp_slider = Scale(self.frame_hough, from_=1.0, to=3.0, resolution=0.1, orient=HORIZONTAL, variable=self.dp_var, command=self.update_filter)
        self.dp_slider.grid(row=0, column=1, padx=5)

        ttk.Label(self.frame_hough, text="minDist:").grid(row=0, column=2, padx=5)
        self.min_dist_slider = Scale(self.frame_hough, from_=1, to=100, orient=HORIZONTAL, variable=self.min_dist_var, command=self.update_filter)
        self.min_dist_slider.grid(row=0, column=3, padx=5)

        ttk.Label(self.frame_hough, text="param1:").grid(row=0, column=4, padx=5)
        self.param1_slider = Scale(self.frame_hough, from_=1, to=100, orient=HORIZONTAL, variable=self.param1_var, command=self.update_filter)
        self.param1_slider.grid(row=0, column=5, padx=5)

        ttk.Label(self.frame_hough, text="param2:").grid(row=0, column=6, padx=5)
        self.param2_slider = Scale(self.frame_hough, from_=1, to=100, orient=HORIZONTAL, variable=self.param2_var, command=self.update_filter)
        self.param2_slider.grid(row=0, column=7, padx=5)

        self.frame_hough.grid_remove()

    def toggle_hough_controls(self, event=None):
        selected_filter = self.filter_combo.get()
        if selected_filter == "Hough Circles":
            self.frame_hough.grid()
            self.kernel_slider.grid_remove()
        else:
            self.frame_hough.grid_remove()
            self.kernel_slider.grid()
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            img = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.display_image(self.original_image, self.canvas_original)
            self.update_filter()

    def update_filter(self, *args):
        if self.original_image is not None:
            filter_name = self.filter_combo.get()
            if filter_name == "Hough Circles":
                self.filtered_image = self.apply_hough_circles()
            else:
                kernel_size = self.kernel_slider.get()
                self.filtered_image = self.filter_map[filter_name](kernel_size)
            self.display_image(self.filtered_image, self.canvas_filtered)

    def apply_blur(self, kernel_size):
        """
        Apply a simple blur filter to the image.
        The blur filter averages pixels over a kernel of the given size.
        """
        return cv2.blur(self.original_image, (kernel_size, kernel_size))

    def apply_gaussian(self, kernel_size):
        """
        Apply a Gaussian blur filter to the image.
        This filter uses a Gaussian kernel to smooth the image.
        """
        return cv2.GaussianBlur(self.original_image, (kernel_size, kernel_size), 0)

    def apply_median(self, kernel_size):
        """
        Apply a median blur filter to the image.
        This filter replaces each pixel's value with the median value of the pixels in the kernel.
        """
        return cv2.medianBlur(self.original_image, kernel_size)
    
    def apply_hpf(self, kernel_size):
        """
        Apply a high-pass filter to the image.
        This filter highlights edges and fine details by subtracting a Gaussian blurred image from the original.
        """
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
        hpf_image = cv2.subtract(gray_image, blurred_image)
        return cv2.cvtColor(hpf_image, cv2.COLOR_GRAY2BGR)
    
    def apply_sobel_edge_detector(self, kernel_size):
        """
        Apply a Sobel edge detection filter to the image.
        The Sobel filter calculates the gradient of the image intensity at each pixel.
        """
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)
        sobel_image = np.sqrt(sobel_x**2 + sobel_y**2)
        return cv2.normalize(sobel_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    def apply_prewitt_edge_detector(self, kernel_size):
        """
        Apply a Prewitt edge detection filter to the image.
        The Prewitt filter is similar to the Sobel filter but uses simpler convolution kernels.
        """
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        prewitt_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        prewitt_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1,
        gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)
        prewitt_image = np.sqrt(prewitt_x**2 + prewitt_y**2)
        return cv2.normalize(prewitt_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def apply_erosion(self, kernel_size):
        """
        Apply an erosion filter to the image.
        The erosion filter erodes the boundaries of the foreground object.
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(self.original_image, kernel, iterations=1)

    def apply_hough_circles(self):
        """
        Apply Hough Circle detection to the image.
        This filter detects circles in the image using the Hough Circle Transform algorithm.
        """
        gray_img = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        gray_img = cv2.medianBlur(gray_img, 5)
        dp = self.dp_var.get()
        min_dist = self.min_dist_var.get()
        param1 = self.param1_var.get()
        param2 = self.param2_var.get()
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=0, maxRadius=0)
        result_img = self.original_image.copy()

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(result_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(result_img, (i[0], i[1]), 2, (0, 0, 255), 3)

        return result_img

    def apply_dilation(self, kernel_size):
        """
        Apply a dilation filter to the image.
        The dilation filter increases the object area and is used to accentuate features.
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(self.original_image, kernel, iterations=1)

    def display_image(self, img, canvas):
        """
        Display an image on a specified Tkinter canvas.
        """
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img_pil)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageFilterApp(root)
    root.mainloop()