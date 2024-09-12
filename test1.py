import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageFont
import cv2
from tkinter import filedialog
from PIL import ImageGrab
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

#root.title("Your Desired Window Title")

# Parameters for preview and export resolutions
PREVIEW_WIDTH = 640  # For preview resolution (e.g., 1280
PREVIEW_HEIGHT = 360  # For preview resolution (e.g., 720

EXPORT_WIDTH = 1920  # For export resolution (e.g., 1920
EXPORT_HEIGHT = 1080  # For export resolution (e.g., 1080


# Function to select an area in an image with a 16:9 aspect ratio using OpenCV
def select_area(image_path):
    # Load the original image
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]

    # Desired maximum size for the selection window (1280x720)
    max_w, max_h = 1280, 720

    # Calculate the scaling factor to fit the image within the selection window while preserving the aspect ratio
    scale_w = max_w / img_w
    scale_h = max_h / img_h
    scale = min(scale_w, scale_h)  # Choose the smaller scaling factor to ensure it fits in the window

    # Resize the image while preserving the aspect ratio
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    img = cv2.resize(img, (new_w, new_h))

    clone = img.copy()
    ref_point = []
    aspect_ratio = 16 / 9

    # Mouse callback function to enforce 16:9 aspect ratio during drawing
    def shape_selection(event, x, y, flags, param):
        nonlocal img, ref_point

        if event == cv2.EVENT_LBUTTONDOWN:
            ref_point = [(x, y)]  # Store the initial point
        elif event == cv2.EVENT_MOUSEMOVE and len(ref_point) == 1:
            img = clone.copy()  # Reset image to remove previous rectangle

            # Get the starting point
            x0, y0 = ref_point[0]

            # Calculate the width and height keeping a 16:9 aspect ratio
            width = abs(x - x0)
            height = int(width / aspect_ratio)

            # Determine direction and adjust coordinates
            if x < x0:
                x1 = x0 - width
            else:
                x1 = x0

            if y < y0:
                y1 = y0 - height
            else:
                y1 = y0

            # Make sure that the drawn rectangle is within the image's boundaries
            x1 = max(0, min(x1, new_w))
            y1 = max(0, min(y1, new_h))
            x2 = min(x1 + width, new_w)
            y2 = min(y1 + height, new_h)

            # Draw the rectangle with the enforced 16:9 aspect ratio
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            # Finalize the rectangle after mouse release
            x0, y0 = ref_point[0]
            x1, y1 = x, y

            # Calculate the width and height while maintaining the aspect ratio
            width = abs(x1 - x0)
            height = int(width / aspect_ratio)

            # Maintain the 16:9 ratio based on the final visual position
            if width / height > aspect_ratio:
                # Adjust width to maintain aspect ratio
                width = int(height * aspect_ratio)
            else:
                # Adjust height to maintain aspect ratio
                height = int(width / aspect_ratio)

            # Determine the direction and recalculate final x1 and y1
            if x1 < x0:
                x1 = x0 - width
            else:
                x1 = x0 + width

            if y1 < y0:
                y1 = y0 - height
            else:
                y1 = y0 + height

            # Ensure the rectangle doesn't exceed image boundaries
            x1 = max(0, min(x1, new_w))
            y1 = max(0, min(y1, new_h))

            ref_point.append((x1, y1))  # Store the final point

            # Draw the final 16:9 rectangle
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
            cv2.imshow("Select Area (16:9 Enforced)", img)

    # Set up the OpenCV window and bind the mouse callback function
    cv2.namedWindow("Select Area (16:9 Enforced)")
    cv2.setMouseCallback("Select Area (16:9 Enforced)", shape_selection)

    while True:
        cv2.imshow("Select Area (16:9 Enforced)", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            img = clone.copy()  # Reset the image
        elif key == 13:  # Enter key to finalize selection
            break

    cv2.destroyAllWindows()

    # Ensure ref_point has valid points before cropping
    if len(ref_point) == 2:
        x0, y0 = ref_point[0]
        x1, y1 = ref_point[1]

        # Ensure that the coordinates are within bounds and order them properly
        x0, x1 = sorted([int(x0), int(x1)])
        y0, y1 = sorted([int(y0), int(y1)])

        # Crop the selected area based on the recalculated rectangle
        cropped_img = clone[y0:y1, x0:x1]

        # Ensure cropped_img is not empty
        if cropped_img.size > 0:
            return Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))  # Convert to PIL image
        else:
            print("Error: Cropped image is empty.")
            return None
    else:
        print("Error: Invalid selection.")
        return None


# Function to ask for user input (for center text)
def get_user_text():
    root = tk.Tk()
    root.withdraw()
    text = simpledialog.askstring("Input", "Enter the text for the center:")
    root.destroy()
    return text


# Function to add white bars with drop shadows
def add_bars_with_shadow(image, bar_thickness, canvas_size=(1920, 1080)):
    shadow_offset = (10, 10)
    shadow_blur_radius = 10

    # Create an image for drawing the shadow and bars
    base = Image.new('RGBA', canvas_size, (255, 255, 255, 0))
    shadow = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

    draw = ImageDraw.Draw(base)
    shadow_draw = ImageDraw.Draw(shadow)

    # Add horizontal and vertical bars
    bar_color = (255, 255, 255, 255)  # white bar
    shadow_color = (0, 0, 0, 150)  # semi-transparent black for shadow

    # Positions for the bars
    center_x = canvas_size[0] // 2
    center_y = canvas_size[1] // 2

    # Draw shadow for the vertical bar
    shadow_draw.rectangle([center_x - bar_thickness // 2 + shadow_offset[0],
                           0 + shadow_offset[1],
                           center_x + bar_thickness // 2 + shadow_offset[0],
                           canvas_size[1] + shadow_offset[1]], fill=shadow_color)

    # Draw shadow for the horizontal bar
    shadow_draw.rectangle([0 + shadow_offset[0],
                           center_y - bar_thickness // 2 + shadow_offset[1],
                           canvas_size[0] + shadow_offset[0],
                           center_y + bar_thickness // 2 + shadow_offset[1]], fill=shadow_color)

    # Apply blur to the shadow for smooth effect
    shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur_radius))

    # Draw the white bars
    draw.rectangle([center_x - bar_thickness // 2, 0, center_x + bar_thickness // 2, canvas_size[1]], fill=bar_color)
    draw.rectangle([0, center_y - bar_thickness // 2, canvas_size[0], center_y + bar_thickness // 2], fill=bar_color)

    # Composite the shadow and the bars onto the main image
    image.paste(shadow, (0, 0), shadow)
    image.paste(base, (0, 0), base)

    return image


# Function to add text box with drop shadow
def add_text_box(image, text, canvas_size=(1920, 1080)):
    shadow_offset = (5, 5)
    shadow_blur_radius = 5
    text_box_padding = 20

    # Create a shadow layer
    shadow = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
    draw_shadow = ImageDraw.Draw(shadow)
    draw_text = ImageDraw.Draw(image)

    # Load the font, fallback to a default font if Arial is unavailable
    try:
        font = ImageFont.truetype("arial.ttf", 48)  # Increased font size to suit 1920x1080 resolution
    except IOError:
        font = ImageFont.load_default()

    # Get text size using the font
    text_w, text_h = draw_text.textbbox((0, 0), text, font=font)[2:]

    # Define text box dimensions
    box_coords = [canvas_size[0] // 2 - text_w // 2 - text_box_padding,
                  canvas_size[1] // 2 - text_h // 2 - text_box_padding,
                  canvas_size[0] // 2 + text_w // 2 + text_box_padding,
                  canvas_size[1] // 2 + text_h // 2 + text_box_padding]

    # Draw the shadow for the text box
    draw_shadow.rectangle([box_coords[0] + shadow_offset[0], box_coords[1] + shadow_offset[1],
                           box_coords[2] + shadow_offset[0], box_coords[3] + shadow_offset[1]],
                           fill=(0, 0, 0, 150))

    # Blur the shadow
    shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur_radius))

    # Composite the shadow onto the image **before** the white box
    image.paste(shadow, (0, 0), shadow)

    # Draw the white box
    draw_text.rectangle(box_coords, fill="white")

    # Draw the text in the center of the box
    draw_text.text((canvas_size[0] // 2, canvas_size[1] // 2), text, font=font, anchor="mm", fill="black")

    return image



# Function to resize and move images within the canvas
class ImageEditor:
    def __init__(self, root, images, text):
        self.root = root
        self.preview_width = PREVIEW_WIDTH
        self.preview_height = PREVIEW_HEIGHT
        self.export_width = EXPORT_WIDTH
        self.export_height = EXPORT_HEIGHT
        self.text = text
        self.images = images

        # Create the canvas for the preview
        self.canvas = tk.Canvas(root, width=self.preview_width, height=self.preview_height)
        self.canvas.pack()

        # Create a blank canvas image for preview and export (preview resolution)
        self.composite_image = Image.new('RGBA', (self.preview_width, self.preview_height), (255, 255, 255, 255))

        # Resize images to fit a 2x2 grid in the preview canvas (scaled to preview resolution)
        self.resized_images = [img.resize((self.preview_width // 2, self.preview_height // 2), Image.LANCZOS) for img in
                               images]
        self.positions = [(0, 0), (self.preview_width // 2, 0), (0, self.preview_height // 2),
                          (self.preview_width // 2, self.preview_height // 2)]

        # Create a slider to adjust the bar thickness
        self.bar_slider = tk.Scale(root, from_=1, to=200, orient=tk.HORIZONTAL, label="Balken Dikte",
                                   command=self.update_image)
        self.bar_slider.pack()
        self.bar_slider.set(10)

        # Create an Export button to export the image in 1080p
        self.export_button = tk.Button(root, text="Export Image", command=self.export_image)
        self.export_button.pack()

        # Initial rendering at preview resolution
        self.update_image(50)

    def update_image(self, bar_thickness):
        # Clear previous image (for preview)
        self.composite_image = Image.new('RGBA', (self.preview_width, self.preview_height), (255, 255, 255, 255))

        # Paste the images onto the composite image (preview resolution)
        for i, img in enumerate(self.resized_images):
            self.composite_image.paste(img, self.positions[i])

        # Add white bars with shadows (scaled to preview resolution)
        self.composite_image = add_bars_with_shadow(self.composite_image, int(bar_thickness),
                                                    (self.preview_width, self.preview_height))

        # Add text box with shadow (scaled to preview resolution)
        self.composite_image = add_text_box(self.composite_image, self.text, (self.preview_width, self.preview_height))

        # Convert the composite image to Tkinter format and display it in preview
        self.tk_image = ImageTk.PhotoImage(self.composite_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def export_image(self):
        # Ask user where to save the exported image, with a default file name
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")],
                                                 title="Save Image As", initialfile="exported_image_1080p.png")

        if file_path:  # Check if the user provided a file path
            # Create an image for export at 1080p resolution
            export_image = Image.new('RGBA', (self.export_width, self.export_height), (255, 255, 255, 255))

            # Resize the images to fit a 2x2 grid in export resolution (e.g., 1920x1080)
            export_resized_images = [img.resize((self.export_width // 2, self.export_height // 2), Image.LANCZOS) for
                                     img in self.images]
            export_positions = [(0, 0), (self.export_width // 2, 0), (0, self.export_height // 2),
                                (self.export_width // 2, self.export_height // 2)]

            # Paste the images onto the export image (1080p resolution)
            for i, img in enumerate(export_resized_images):
                export_image.paste(img, export_positions[i])

            # Add white bars with shadows (1080p resolution)
            export_image = add_bars_with_shadow(export_image, int(self.bar_slider.get()),
                                                (self.export_width, self.export_height))

            # Add text box with shadow (1080p resolution)
            export_image = add_text_box(export_image, self.text, (self.export_width, self.export_height))

            # Save the final image at the user-specified location
            export_image.save(file_path)
            print(f"Image exported to: {file_path}")


def main():
    root = tk.Tk()
    root.title("combine images")  # Set the main window title

    # Ask user to select 4 images
    image_paths = []
    for i in range(4):
        image_paths.append(filedialog.askopenfilename(title=f"Select Image {i+1}"))

    # Select area from each image
    images = [select_area(path) for path in image_paths]

    # Get text for the center
    center_text = get_user_text()

    # Start the image editor
    editor = ImageEditor(root, images, center_text)
    root.mainloop()



if __name__ == "__main__":
    main()
