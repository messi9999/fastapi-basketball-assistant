from PIL import Image

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("model/best.pt")

# Run inference on 'bus.jpg'
results = model(["testdata/1.jpg", "testdata/2.jpg", "testdata/3.jpg"])  # results list

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")