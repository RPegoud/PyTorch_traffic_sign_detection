new_labels = {}
for idx in range(9):
    new_labels[idx] = 1 # speed limitation
for idx in range(18, 32):
    new_labels[idx] = 2 # danger
for idx in range(33, 41):
    new_labels[idx] = 3 # change in direction
for idx in [32, 41, 42]:
    new_labels[idx] = 4 # end of limitations
for idx in [9, 10, 15, 16, 17]:
    new_labels[idx] = 5 # no circulation
for idx in range(11, 14):
    new_labels[idx] = 6 # priority
new_labels[14] = 7 # stop

new_labels_to_category = {
    1:'Speed limitation',
    2:'Danger',
    3:'Change in direction',
    4:'End of limitations',
    5:'No circulation',
    6:'Priority',
    7:'Stop'
}