import os

from matplotlib import pyplot as plt
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class TextHandler(HandlerBase):
    def create_artists(self, legend, text, xdescent, ydescent, width, height, fontsize, trans):
        tx = Text(width / 2., height / 2, text, fontsize=fontsize, ha="center", va="center", fontweight="bold")
        return [tx]


def draw_confusion_matrix(correct_arr, predicted_arr, classes, save_fig=True, result_path='results', font_size=11, show_diagram=True):
    cm = confusion_matrix(correct_arr, predicted_arr, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    Legend.update_default_handler_map({str: TextHandler()})
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': font_size})

    disp.plot(cmap='jet', ax=ax)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    ax.legend(handles=[str(i) for i in range(len(classes))], labels=[c for id, c in enumerate(classes)],
              loc='upper left', bbox_to_anchor=(1.04, 1))

    if save_fig:
        # fig.set_size_inches((8.5, 11), forward=False)
        fig = plt.gcf()
        fig.set_size_inches((18, 8), forward=False)
        fig.savefig(os.path.join(result_path, 'evaluate.png'))

    if show_diagram:
        plt.show()
