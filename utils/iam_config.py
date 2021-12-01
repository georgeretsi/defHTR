# iam configuration 
# HARDCODED !! CHANGE THIS !!

#iam_root = '/media/ncsr/bee4cbda-e313-4acf-9bc8-817a69ad98ae/IAM'
iam_root = 'datasets/IAM'

trainset_file = '{}/set_split/trainset.txt'.format(iam_root)
testset_file = '{}/set_split/testset.txt'.format(iam_root)

line_file = '{}/ascii/lines.txt'.format(iam_root)
word_file = '{}/ascii/words.txt'.format(iam_root)

word_path = '{}/words'.format(iam_root)
line_path = '{}/lines'.format(iam_root)

stopwords_path = '{}/iam-stopwords'.format(iam_root)

dataset_path = './saved_datasets'
