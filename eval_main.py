import os
from eval.eval_yolo import evaluate_yolo_models, print_yolo_metrics
from eval.eval_ocrs import evaluate_ocrs
from eval.eval_pipeline import evaluate_pipeline

def clear_terminal():
    os.system('cls' if os.name in ('nt', 'dos') else 'clear')

if __name__ == "__main__":
    yolo_results = evaluate_yolo_models()
    clear_terminal()
    for result in yolo_results:
        print_yolo_metrics(result['metrics'], result['model'])
    input("Press Enter to start OCR evaluation...")

    ocr_results = evaluate_ocrs()
    clear_terminal()
    for name, (char_acc, plate_acc) in ocr_results.items():
        print(f"{name}: Char accuracy: {char_acc:.4f}, Plate accuracy: {plate_acc:.4f}")
    input("Press Enter to start pipeline evaluation...")

    pipeline_results = evaluate_pipeline()
    clear_terminal()
    for result in pipeline_results:
        print(f"{result['yolo_model']} + {result['enhancer']} + {result['ocr']}: "
              f"Plate Acc = {result['plate_accuracy']:.2f}%, Char Acc = {result['char_accuracy']:.2f}%")
