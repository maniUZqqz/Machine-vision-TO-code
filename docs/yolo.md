# YOLO (optional)

The v2 converter does **not** need a neural detector: regions, text lines, photos, rules and
controls come from pixel analysis (see `mv2code/segment.py`, `textdet.py`).  The v1 code path
(`vision_engine/detection/yolo_detector.py`, `ml_models/yolo_ui/`) is still in the repository and
keeps working without `ultralytics` installed - it simply stays off.

* **Install**: `pip install "mv2code[yolo]"` (pulls `ultralytics`, which pulls torch - do not do this on a small server).
* **Training happens on a GPU machine, never on the server.**  On any machine with a CUDA GPU:

  ```bash
  git clone https://github.com/maniUZqqz/Machine-vision-TO-code && cd Machine-vision-TO-code
  pip install -e ".[yolo,web]"
  python manage.py train_yolo --epochs 60 --batch 16 --synthetic-count 4000   # synthetic UI data from ml_models/yolo_ui
  # result: ml_models/yolo_ui/weights/best.pt
  ```

* Copy `best.pt` next to the code that should use it (`ml_models/yolo_ui/weights/best.pt`).  If the
  file is absent the detector reports `available = False` and the classical pipeline runs alone.
* The benchmark (`python -m bench run`) never loads YOLO; numbers in the README are classical-only.
