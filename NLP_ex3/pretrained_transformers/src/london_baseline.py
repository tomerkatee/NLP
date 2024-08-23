# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils

filename = 'birth_dev.tsv'

with open(filename, 'r') as file:
    line_cnt = len(file.readlines())

london_predictions = ['London' for _ in range(line_cnt)]

total, correct = utils.evaluate_places(filename, london_predictions)
print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))

