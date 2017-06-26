"""
This file provides a helper script to evaluate Mean Average Precision.
"""
from os.path import join


def read_class(filename):
    """
    Reads a dataset split and returns a dictionary like {img_name: class}.

    :param filename: the file containing the split.
    :return: a dictionary like {img_name: class}.
    """
    with open(filename) as f:
        lines = f.readlines()
        ret = {' '.join(l.split()[:-1]): l.split()[-1] for l in lines}

    return ret

train_class = read_class('train_list.txt')
test_class = read_class('test_list.txt')

def eval_map(prediction_folder):
    """
    Evaluates the mean average precision on the test set.

    :param prediction_folder: the folder holding predictions.
    :return: a dictionary like {class: map_score}.
    """
    # average precisions structure
    aps = {
        'animals': [],
        'food': [],
        'landscapes': [],
        'people': [],
        'tools': []
    }
    # for every test query
    for img, cl in test_class.iteritems():
        #print "[[{}]] -> {}".format(img,cl)
        # read the predictions
        try:
            with open(join(prediction_folder, img + '.txt')) as f:
                sorted_list = f.readlines()
                sorted_list = [f.strip() for f in sorted_list]
        except:
            print 'immagine non trovata durante la predizione'
            continue
        count = 1.
        tp = 0.
        precisions = []

        # for each result
        for res in sorted_list:
            try:
                if train_class[res] == cl:  # true positive
                    tp += 1
                    precisions.append(tp / count)
                # Il count va messo qui (dopo la riga che puo' far scattare l'eccezione, cioe' quella con
                # train_class[res]) altrimenti si incrementa il contatore anche quando la riga contiene un'immagine che
                # non e' presente nel training set fornito da Davide, diminuendo la MAP.
                # Di conseguenza count parte da 1. invece che da 0.
                count += 1
            except KeyError:
                print 'non ho {} nella train list'.format(res)
        assert len(precisions) == tp

        # update the average precisions for the class
        if len(precisions)==0:
            #print precisions,'precision vuota'
            aps[cl].append(0)
        else:
            aps[cl].append(sum(precisions)/len(precisions))

    # turn list of aps into maps
    maps = {key: sum(value)/len(value) for key, value in aps.iteritems()}
    return maps

models = ['hist_sift','hist_color','resnet50','vgg16','vgg19','resnet50_cl','vgg16_cl','vgg19_cl','inception_resnet_v2','inception_resnet_v2_cl']
#models = ['hist_sift']

if __name__ == '__main__':
    text = ''
    for model in models:
        maps = eval_map(prediction_folder='predictions_'+model)

        # print
        try:
            # fancy print
            from prettytable import PrettyTable
            table = PrettyTable()
            table.field_names = [cl for cl in maps] + ['mean']

            map_values = [value for _, value in maps.iteritems()]
            table.add_row(map_values + [sum(map_values) / len(map_values)])
            #print model
            print table

            text += model + '\n' + table.get_string() + '\n\n'
        except Exception as e:
            print e.message
            # sad print
            print maps
    with open('FINAL_RESULT.txt', 'w') as file:
        file.write(text)

