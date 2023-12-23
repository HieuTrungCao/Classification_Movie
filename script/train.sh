python train.py --path_data './data/dataset' --batch_size 256 --save_path "/content/save_model" \
  --threshold 0.6 --iter_print 4 --epoch 40 --pretrained True  --max_length 30 --use_title True\
  --notes "Resnet 50 + distilBert + data aug" --nlp_model "distilbert-base-uncased" --get_year True