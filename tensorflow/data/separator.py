#!/usr/bin/env python3.5
# coding: utf-8

import os
import numpy as np
import argparse
from math import floor


def main(args):
  # class foo(object):
  #   pass
  # args = foo()
  # args.ref='raw/training2017/REFERENCE.csv'
  annot_lines = open(args.ref, 'r').read().splitlines()
  np.random.shuffle(annot_lines)
  annot_dict = {s: s.split(',')[1] for s in annot_lines}
  index_dict = {'N': [], 'A': [], 'O': [], '~': []}
  for idx, line in enumerate(annot_lines):
    index_dict[annot_dict[line]].append(idx)

  TRAIN = args.train / 100.
  VAL = args.val / 100.
  print('Sample in class/set:')
  print('\tTotal,\tTrain,\tValid,\tTest')
  for x in index_dict.items():
    l = len(x[1])
    hist = (x[0], l, floor(l * TRAIN), floor(l * VAL),
        l - floor((TRAIN + VAL) * l))
    print('%s,\t%d,\t%d,\t%d\t%d' % hist)

  def fp(x): return os.path.normpath(os.path.dirname(args.ref) + '/' + x)
  train_reference = open(fp('TRAIN.csv'), 'w')
  validation_reference = open(fp('VALIDATION.csv'), 'w')
  test_reference = open(fp('TEST.csv'), 'w')

  for idxs in index_dict.values():
    l = len(idxs)
    train_reference.writelines(
      '%s\n' % annot_lines[i] for i in idxs[:floor(l * TRAIN)])
    validation_reference.writelines(
      '%s\n' % annot_lines[i] for i in idxs[floor(l * TRAIN):floor(l * (TRAIN + VAL))])
    test_reference.writelines(
      '%s\n' % annot_lines[i] for i in idxs[floor(l * (TRAIN + VAL)):])
  print('References written succesfully to:')
  print(train_reference.name,
      validation_reference.name,
      test_reference.name,
      sep='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ref', help='location of reference file',
        default='./raw/training2017/REFERENCE.csv')
    parser.add_argument(
        '--train', help='percents of files from `--ref` to keep in train set',
        type=int, default=80)
    parser.add_argument(
        '--val', help='percents of files from `--ref` to keep in val set',
        type=int, default=10)
    args = parser.parse_args()
    main(args)
