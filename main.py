import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from bilstm_evaluation import evluation
from music_data_preprocess import X_music, X_shuffled_music, Y_music, note2id
from language_data_preprocess import X_language, X_shuffled_language, Y_language, word2id

def main():

    args = sys.argv

    if args[1] == 'Language':
        X_language_train_valid, X_language_test, y_language_train_valid, y_language_test = train_test_split(X_language, Y_language, test_size=0.1, random_state=23)
        X_language_train, X_language_valid, y_language_train, y_language_valid = train_test_split(X_language_train_valid, y_language_train_valid, test_size=0.1, random_state=26)
        print("Experiment 0: Only using language data to train")
        mode = -1
        eval = evluation(word2id, mode)
        eval.train(X_language_train, y_language_train, X_language_valid, y_language_valid, word2id, finetune=False)
        y_pred, y_true = eval.test(X_language_test, y_language_test, word2id) 

    elif args[1] == 'Music':
        X_music_train_valid, X_music_test, y_music_train_valid, y_music_test = train_test_split(X_music, Y_music, test_size=0.1, random_state=23)
        X_music_train, X_music_valid, y_music_train, y_music_valid = train_test_split(X_music_train_valid, y_music_train_valid, test_size=0.1, random_state=26)
        print("Experiment 1: Only using music data to train")
        mode = 0
        eval = evluation(note2id, mode)
        eval.train(X_music_train, y_music_train, X_music_valid, y_music_valid, note2id, finetune=False)
        y_pred, y_true = eval.test(X_music_test, y_music_test, note2id)

    elif args[1] == 'Language_Music':
        X_music_train_valid, X_music_test, y_music_train_valid, y_music_test = train_test_split(X_music, Y_music, test_size=0.1, random_state=23)
        X_music_train, X_music_valid, y_music_train, y_music_valid = train_test_split(X_music_train_valid, y_music_train_valid, test_size=0.1, random_state=26)
        X_language_train, X_language_valid, y_language_train, y_language_valid = train_test_split(X_language, Y_language, test_size=0.1, random_state=26)
        print("Experiment 2: Using language data to pretrain and music data to finetune")
        mode = 1
        eval = evluation(word2id, mode)
        print("Begin to pretrain")
        eval.train(X_language_train, y_language_train, X_language_valid, y_language_valid, word2id, finetune=False)
        print("Begin to finetune")
        eval.train(X_music_train, y_music_train, X_music_valid, y_music_valid, note2id, finetune=True)
        y_pred, y_true = eval.test(X_music_test, y_music_test, note2id)

    elif args[1] == 'RandomLanguage_Music':
        X_music_train_valid, X_music_test, y_music_train_valid, y_music_test = train_test_split(X_music, Y_music, test_size=0.1, random_state=23)
        X_music_train, X_music_valid, y_music_train, y_music_valid = train_test_split(X_music_train_valid, y_music_train_valid, test_size=0.1, random_state=26)
        X_shuffled_train, X_shuffled_valid, y_language_train, y_language_valid = train_test_split(X_shuffled_language, Y_language, test_size=0.1, random_state=26)
        print("Experiment 3: Using shuffled language data to pretrain and music data to finetune")
        mode = 2
        eval = evluation(word2id, mode)
        print("Begin to pretrain")
        eval.train(X_shuffled_train, y_language_train, X_shuffled_valid, y_language_valid, word2id, finetune=False)
        print("Begin to finetune")
        eval.train(X_music_train, y_music_train, X_music_valid, y_music_valid, note2id, finetune=True)
        y_pred, y_true = eval.test(X_music_test, y_music_test, note2id)

    elif args[1] == 'Music_Language':
        X_language_train_valid, X_language_test, y_language_train_valid, y_language_test = train_test_split(X_language, Y_language, test_size=0.1, random_state=23)
        X_language_train, X_language_valid, y_language_train, y_language_valid = train_test_split(X_language_train_valid, y_language_train_valid, test_size=0.1, random_state=26)
        X_music_train, X_music_valid, y_music_train, y_music_valid = train_test_split(X_music, Y_music, test_size=0.1, random_state=26)
        print("Experiment 4: Using music data to pretrain and language data to finetune")
        mode = 3
        eval = evluation(word2id, mode)
        print("Begin to pretrain")
        eval.train(X_music_train, y_music_train, X_music_valid, y_music_valid, note2id, finetune=False)
        print("Begin to finetune")
        eval.train(X_language_train, y_language_train, X_language_valid, y_language_valid, word2id, finetune=True)
        y_pred, y_true = eval.test(X_language_test, y_language_test, word2id)

    elif args[1] == 'RandomMusic_Language':
        X_language_train_valid, X_language_test, y_language_train_valid, y_language_test = train_test_split(X_language, Y_language, test_size=0.1, random_state=23)
        X_language_train, X_language_valid, y_language_train, y_language_valid = train_test_split(X_language_train_valid, y_language_train_valid, test_size=0.1, random_state=26)
        X_shuffled_train, X_shuffled_valid, y_music_train, y_music_valid = train_test_split(X_shuffled_music, Y_music, test_size=0.1, random_state=26)
        print("Experiment 5: Using shuffled music data to pretrain and language data to finetune")
        mode = 4
        eval = evluation(word2id, mode)
        print("Begin to pretrain")
        eval.train(X_shuffled_train, y_music_train, X_shuffled_valid, y_music_valid, note2id, finetune=False)
        print("Begin to finetune")
        eval.train(X_language_train, y_language_train, X_language_valid, y_language_valid, word2id, finetune=True)
        y_pred, y_true = eval.test(X_language_test, y_language_test, word2id)


    # flatten list
    y_pred_list = [item for sublist in y_pred for item in sublist]
    y_true_list = [int(item) for sublist in y_true for item in sublist]

    # calculate f1 score and confusion matrix
    f1score = precision_recall_fscore_support(y_true_list, y_pred_list, average='binary')[2]
    c_matrix = confusion_matrix(y_true_list, y_pred_list, labels=[0, 1])

    print('The f1-score is {}'.format(f1score))
    print('The confusion matrix is \n {}'.format(c_matrix))

if __name__ == "__main__":
    main()
