#! /usr/bin/env ipython3
import sys
import pandas as pd


def main(argv):
    answers = 'answers.txt'
    reference = 'data/raw/training2017/REFERENCE.csv'
    if len(argv) == 2:
        answers = argv[1]
    if len(argv) == 3:
        answers = argv[1]
        reference = argv[2]
    preds = pd.read_csv(answers, names=['name', 'prediction'])
    truth = pd.read_csv(reference, names=['name', 'label'])
    merged = preds.sort_values('name').merge(
            truth.sort_values('name'), on='name')

    def f1(class_char):
        all_pred = merged[merged['prediction'] == class_char]
        all_label = merged[merged['label'] == class_char]
        prec = merged[merged['prediction'] == merged['label']]
        prec = prec[prec["label"] == class_char]
        prec.count()
        all_pred.count() + all_label.count()
        2 * prec.count() / (all_pred.count() + all_label.count())
        return 2 * (prec.count() / (all_pred.count() + all_label.count()))[0]

    f1_N = f1('N')
    f1_O = f1('O')
    f1_A = f1('A')

    print('-' * 10)
    print('Normal: %.2f' % f1_N)
    print('Other: %.2f' % f1_O)
    print('AF: %.2f' % f1_A)
    print('-' * 10)
    print('Overall score: %.2f' % ((f1_N + f1_O + f1_A) / 3))


if __name__ == '__main__':
    main(sys.argv)
