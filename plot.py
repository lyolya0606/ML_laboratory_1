import pandas as pd
import matplotlib.pyplot as plt

# Названия файлов и их метки для легенды
files = {
    "Adam_train": "Adam Train",
    "Adam_valid": "Adam Valid",
    "AdamW_train": "AdamW Train",
    "AdamW_valid": "AdamW Valid"
}

# Создаем пустой словарь для хранения данных
data = {}

# Читаем данные из файлов
for filename, label in files.items():
    df = pd.read_csv(f"./points/{filename}.csv", header=None, names=["Loss", "Acc@1", "Acc@5"])
    data[label] = df  # Сохраняем DataFrame в словарь


# Функция для построения графиков
def plot_metric(metric, ylabel, filename):
    plt.figure(figsize=(8, 5))
    for label, df in data.items():
        plt.plot(df[metric], label=label)

    plt.xlabel("Эпохи")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} по эпохам")
    plt.legend()
    plt.grid(True)
    plt.savefig('./chart_png/' + filename)  # Сохраняем в файл
    plt.close()  # Закрываем график, чтобы не перегружать память

# Вызываем функции с сохранением
plot_metric("Loss", "Loss", "loss_plot.png")
plot_metric("Acc@1", "Accuracy top 1 (%)", "acc1_plot.png")
plot_metric("Acc@5", "Accuracy top 5 (%)", "acc5_plot.png")