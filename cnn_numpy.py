import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# ---------- Helper: Load and preprocess images ----------
def load_data(folder_path, img_size=(32, 32)):
    X, y, labels = [], [], []
    for label_idx, label_name in enumerate(sorted(os.listdir(folder_path))):
        labels.append(label_name)
        label_folder = os.path.join(folder_path, label_name)
        for img_name in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            X.append(img / 255.0)
            y.append(label_idx)
    return np.array(X), np.array(y), labels

# ---------- Convolution Layer ----------
def convolve(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros_like(image)
    pad = kernel.shape[0] // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), 'constant')
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            region = padded_image[y:y+kernel.shape[0], x:x+kernel.shape[1]]
            output[y, x] = np.sum(region * kernel)
    return output

# ---------- ReLU ----------
def relu(x):
    return np.maximum(0, x)

# ---------- Max Pooling ----------
def max_pooling(feature_map, size=2, stride=2):
    out_h = (feature_map.shape[0] - size)//stride + 1
    out_w = (feature_map.shape[1] - size)//stride + 1
    pooled = np.zeros((out_h, out_w))
    for y in range(0, feature_map.shape[0]-size+1, stride):
        for x in range(0, feature_map.shape[1]-size+1, stride):
            pooled[y//stride, x//stride] = np.max(feature_map[y:y+size, x:x+size])
    return pooled

# ---------- Fully Connected Layer ----------
def fully_connected(flat_input, weights, bias):
    return np.dot(flat_input, weights) + bias

# ---------- Softmax ----------
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

# ---------- Cross-Entropy Loss ----------
def cross_entropy_loss(pred_probs, true_idx):
    return -np.log(pred_probs[true_idx] + 1e-9)

# ---------- Load Dataset ----------
X, y, labels = load_data("dataset")
print("Data loaded:", X.shape, "Classes:", labels)

# Use small dataset for demonstration
X = X[:10]
y = y[:10]

# ---------- Define Kernels ----------
edge_kernel = np.array([[1,0,-1],
                        [1,0,-1],
                        [1,0,-1]])

# ---------- Determine Flattened Size Dynamically ----------
sample_conv = convolve(X[0], edge_kernel)
sample_relu = relu(sample_conv)
sample_pooled = max_pooling(sample_relu)
flat_size = sample_pooled.flatten().shape[0]

# ---------- Initialize Dense Layer ----------
num_classes = len(labels)
weights = np.random.randn(flat_size, num_classes) * 0.01
bias = np.zeros(num_classes)
lr = 0.1  # learning rate

# ---------- Training Loop ----------
def train_model():
    global weights, bias
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        for i, img in enumerate(X):
            conv_output = convolve(img, edge_kernel)
            relu_output = relu(conv_output)
            pooled_output = max_pooling(relu_output)
            flat = pooled_output.flatten()
            output = fully_connected(flat, weights, bias)
            probs = softmax(output)

            loss = -np.log(probs[y[i]] + 1e-9)
            total_loss += loss

            grad_output = probs.copy()
            grad_output[y[i]] -= 1
            grad_w = np.outer(flat, grad_output)
            grad_b = grad_output

            weights -= 0.1 * grad_w
            bias -= 0.1 * grad_b

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

# Train once when imported
train_model()

# ---------- Testing / Predictions ----------
print("\nPredictions after training:")
for i, img in enumerate(X):
    conv_output = convolve(img, edge_kernel)
    relu_output = relu(conv_output)
    pooled_output = max_pooling(relu_output)
    flat = pooled_output.flatten()
    output = fully_connected(flat, weights, bias)
    probs = softmax(output)
    pred = np.argmax(probs)
    print(f"Image {i+1} | Actual: {labels[y[i]]} | Predicted: {labels[pred]}")

# ---------- Visualize ----------
plt.imshow(X[0], cmap='gray')
plt.title("Sample Input Image")
plt.show()



# ---------- Testing on a New Image ----------
def predict_image(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0

    # Forward pass
    conv_output = convolve(img, edge_kernel)
    relu_output = relu(conv_output)
    pooled_output = max_pooling(relu_output)
    flat = pooled_output.flatten()
    output = fully_connected(flat, weights, bias)
    probs = softmax(output)

    # Get prediction
    pred_idx = np.argmax(probs)
    pred_label = labels[pred_idx]
    
    print(f"Predicted ASL Letter: {pred_label}")
    
    # Visualize
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {pred_label}")
    plt.show()

# ---------- Example Usage ----------
# Replace with your test image path
#test_image_path = "test1.jpg"
#predict_image(test_image_path)

