#PROJECT CODES
# Apply high-pass filter (HPF) to enhance edges by adding a slider for kernel size selection
def apply_hpf(self):
    # Add a slider for selecting the kernel size (range: 1 to 20, initial value: 5)
    # Call the update_hpf function when the slider value changes
    self.add_slider("Kernel Size", 1, 20, 5, self.update_hpf)

# Update function for high-pass filter
def update_hpf(self, event):
    # Get the selected kernel size from the slider
    kernel_size = int(event.widget.get())
    # Convert the original image to grayscale
    gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to the grayscale image with the selected kernel size
    blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    # Subtract the blurred image from the grayscale image to obtain high-pass filtered image
    hpf_image = cv2.subtract(gray_image, blurred_image)
    # Convert the high-pass filtered image to BGR format and update the image display
    self.update_image(cv2.cvtColor(hpf_image, cv2.COLOR_GRAY2BGR))

# Apply mean filter to smooth the image by adding a slider for kernel size selection
def apply_mean_filter(self):
    # Add a slider for selecting the kernel size (range: 1 to 20, initial value: 5)
    # Call the update_mean_filter function when the slider value changes
    self.add_slider("Kernel Size", 1, 20, 5, self.update_mean_filter)

# Update function for mean filter
def update_mean_filter(self, event):
    # Get the selected kernel size from the slider
    kernel_size = int(event.widget.get())
    # Apply mean filter to the original image with the selected kernel size
    mean_image = cv2.blur(self.original_image, (kernel_size, kernel_size))
    # Update the image display with the mean filtered image
    self.update_image(mean_image)

# Apply median filter to remove noise by adding a slider for kernel size selection
def apply_median_filter(self):
    # Add a slider for selecting the kernel size (odd values only, range: 3 to 21, initial value: 3)
    # Call the update_median_filter function when the slider value changes
    self.add_slider("Kernel Size (Odd)", 3, 21, 3, self.update_median_filter)

# Update function for median filter
def update_median_filter(self, event):
    # Get the selected kernel size from the slider
    kernel_size = int(event.widget.get())
    # Ensure the kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Apply median filter to the original image with the selected kernel size
    median_image = cv2.medianBlur(self.original_image, kernel_size)
    # Update the image display with the median filtered image
    self.update_image(median_image)


# Apply opening operation to the image by adding a slider for kernel size selection
def apply_open(self):
    # Add a slider for selecting the kernel size (range: 1 to 20, initial value: 5)
    # Call the update_open function when the slider value changes
    self.add_slider("Kernel Size", 1, 20, 5, self.update_open)

# Update function for opening operation
def update_open(self, event):
    # Get the selected kernel size from the slider
    kernel_size = int(event.widget.get())
    # Create a kernel for opening
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Perform opening on the original image
    open_image = cv2.morphologyEx(self.original_image, cv2.MORPH_OPEN, kernel)
    # Update the image display with the opened image
    self.update_image(open_image)

# Apply closing operation to the image by adding a slider for kernel size selection
def apply_close(self):
    # Add a slider for selecting the kernel size (range: 1 to 20, initial value: 5)
    # Call the update_close function when the slider value changes
    self.add_slider("Kernel Size", 1, 20, 5, self.update_close)

# Update function for closing operation
def update_close(self, event):
    # Get the selected kernel size from the slider
    kernel_size = int(event.widget.get())
    # Create a kernel for closing
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Perform closing on the original image
    close_image = cv2.morphologyEx(self.original_image, cv2.MORPH_CLOSE, kernel)
    # Update the image display with the closed image
    self.update_image(close_image)

# Apply Hough circle transform to detect circles in the image
def apply_hough_circle_transform(self):
    # Convert the original image to grayscale
    gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
    # Detect circles using Hough circle transform with specified parameters
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
    # Check if any circles are detected
    if circles is not None:
        # Convert the circle parameters to integer
        circles = np.uint16(np.around(circles))
        # Create a copy of the original image for drawing circles
        hough_image = self.original_image.copy()
        # Draw detected circles on the image
        for i in circles[0, :]:
            cv2.circle(hough_image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Draw the outer circle
            cv2.circle(hough_image, (i[0], i[1]), 2, (0, 0, 255), 3)       # Draw the center of the circle
        # Update the image display with the circles drawn
        self.update_image(hough_image)
