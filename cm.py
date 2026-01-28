import tkinter as tk

# ====== CLASSES AND SAMPLE CONFUSION MATRIX ======
classes = [
    "Target Spot",
    "Spider Mites",
    "Yellow Leaf Curl",
    "Septoria",
    "Late Blight",
    "Early Blight",
    "Healthy"
]

# Example 7x7 confusion matrix (change numbers as needed)
confusion_matrix = [
    [12, 1, 0, 0, 0, 0, 0],
    [0, 14, 1, 0, 0, 0, 0],
    [0, 0, 13, 0, 1, 0, 0],
    [0, 0, 0, 15, 0, 0, 0],
    [0, 0, 0, 1, 14, 0, 0],
    [0, 0, 0, 0, 0, 16, 0],
    [0, 0, 0, 0, 0, 0, 18]
]

num_classes = len(classes)

# ====== CALCULATE METRICS ======
# Accuracy
correct = sum(confusion_matrix[i][i] for i in range(num_classes))
total = sum(sum(row) for row in confusion_matrix)
accuracy = correct / total

# Precision, Recall, F1 per class
metrics = []
for i in range(num_classes):
    TP = confusion_matrix[i][i]
    FP = sum(confusion_matrix[r][i] for r in range(num_classes)) - TP
    FN = sum(confusion_matrix[i]) - TP

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    metrics.append((precision, recall, f1))

# ====== TKINTER GUI ======
root = tk.Tk()
root.title("Tomato Disease Detection - Confusion Matrix & Metrics")
root.geometry("1200x650")

title = tk.Label(root, text="Tomato Disease Detection\nConfusion Matrix & Metrics",
                 font=("Arial", 16, "bold"))
title.pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=10)

# ====== TABLE HEADER ======
tk.Label(frame, text="Actual \\ Pred", borderwidth=1, relief="solid",
         width=15, bg="lightgray").grid(row=0, column=0)

for j, cls in enumerate(classes):
    tk.Label(frame, text=cls, borderwidth=1, relief="solid",
             width=15, bg="lightgray").grid(row=0, column=j + 1)

# ====== TABLE BODY ======
for i in range(num_classes):
    tk.Label(frame, text=classes[i], borderwidth=1, relief="solid",
             width=15, bg="lightgray").grid(row=i + 1, column=0)

    for j in range(num_classes):
        bg_color = "lightgreen" if i == j else "white"
        tk.Label(frame, text=str(confusion_matrix[i][j]),
                 borderwidth=1, relief="solid", width=15, bg=bg_color)\
            .grid(row=i + 1, column=j + 1)

# ====== METRICS DISPLAY ======
metrics_frame = tk.Frame(root)
metrics_frame.pack(pady=10)

header = tk.Label(metrics_frame, text=f"{'Class':<20}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}",
                  font=("Arial", 12, "bold"))
header.pack()

for i, (precision, recall, f1) in enumerate(metrics):
    text = f"{classes[i]:<20}{precision:<12.4f}{recall:<12.4f}{f1:<12.4f}"
    tk.Label(metrics_frame, text=text, font=("Arial", 12)).pack(anchor="w")

# ====== ACCURACY ======
acc_label = tk.Label(root, text=f"Overall Accuracy: {accuracy:.4f}",
                     font=("Arial", 14, "bold"), fg="green")
acc_label.pack(pady=15)

root.mainloop()
