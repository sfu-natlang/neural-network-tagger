
import codecs
import utils

if __name__ == '__main__':

  accs = []
  correct_preds, total_correct, total_preds = 0., 0., 0.
  file_input = codecs.open('005_test.txt', 'r', 'UTF-8')
  format_len = 8
  predictions = []
  gold_labels = []
  count = 0
  for cur_line in file_input:
    cur_line = cur_line.strip()
    entity = cur_line.split()
    if len(entity) == format_len:
      predictions.append(entity[-2])
      gold_labels.append(entity[-1])
    else:
      lab_chunks = set(utils.get_chunks(gold_labels))
      lab_pred_chunks = set(utils.get_chunks(predictions))
      correct_preds += len(lab_chunks & lab_pred_chunks)
      total_preds += len(lab_pred_chunks)
      total_correct += len(lab_chunks)
      gold_labels = []
      predictions = []
  p = correct_preds / total_preds if correct_preds > 0 else 0
  r = correct_preds / total_correct if correct_preds > 0 else 0
  f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
  print f1
  file_input.close()
  
