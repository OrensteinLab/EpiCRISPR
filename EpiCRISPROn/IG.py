import keras
import logomaker
import tensorflow as tf
from matplotlib import pyplot as plt
from EpiCRISPR import *

"""In this file we create the saliency map for feature importance"""


# Averages the values of methylation
def avgEpi(epigenetisMarker):
    avgf = 0
    for i in range(len(epigenetisMarker)):
        epi = epigenetisMarker[i].split(',')
        avgf += math.fsum(list(map(float, epi)))
    avgf /= (len(epigenetisMarker) * 32)
    return avgf


# Returns the gradients of the model after training
def get_gradients(model, input_data):
    input_data = tf.convert_to_tensor(input_data)
    with tf.GradientTape() as tape:
        tape.watch(input_data)
        preds = model(input_data)
    grads = tape.gradient(preds, input_data)
    return grads


# Preform gradient ascent and visualize of saliency map
def saliency_map(model, epigenetics):
    x1 = np.ones(shape=(32, 4)) * 0.25  # Initial input matrix
    x1[21:23] = [0, 0, 1, 0]  # GG
    x2 = np.ones(shape=(1, 11))
    x2[:, 10] = 100  # CRISPROn
    x2[:, 3] = avgEpi(epigenetics)  # methylation
    x1 = x1.reshape(1, -1)
    x = np.concatenate((x1, x2), axis=1)
    lr = 0.1
    for i in range(10000):
        grads = get_gradients(model,  x)
        x += grads * lr
    temp = x.numpy()
    seq = temp[0, :-11]
    seq = seq.reshape(32, 4)
    seqDF = pd.DataFrame(seq, columns=['A', 'C', 'G', 'T'])
    seqDF = seqDF.div(seqDF.sum(axis=0), axis=1)
    save_logo(seqDF)
    epi = x[:, -11:].numpy()
    epi[:, 10] /= 100
    visualize_integrated_gradients(seq, 1)
    visualize_integrated_gradients(epi, 0)


# Saliency map visualization
def visualize_integrated_gradients(integrated_gradients, segFlag):
    colors = ['white', 'white', 'white', 'black', 'white', 'white', 'black', 'white', 'black', 'white', 'white']
    if segFlag:
        integrated_gradients = np.flip(np.rot90(integrated_gradients, k=-1), axis=1)
    else:
        num_rows, num_cols = integrated_gradients.shape
        for i in range(num_rows):
            for j in range(num_cols):
                color = colors[j]
                plt.text(j, i, f'{integrated_gradients[i, j]:.2f}', ha='center', va='center', color=color, fontsize=8)
    plt.imshow(integrated_gradients, cmap='Blues', interpolation='nearest', vmin=-2, vmax=2)
    plt.colorbar(label='Feature importance')  # Add color bar
    plt.axis('off')
    plt.show()


# Create the sequence logo according to gradient ascent
def save_logo(df):
    IG_logo = logomaker.Logo(df)
    IG_logo.ax.set_xticks(range(32))
    IG_logo.ax.set_xticklabels(np.arange(1, 33), fontsize=12)
    IG_logo.ax.set_ylabel('Importance score', fontsize=14)
    plt.show()


if __name__ == '__main__':
    reconstructed_model = keras.models.load_model("EpiCRISPR_model.keras")
    e = pd.read_csv('epigenetics/T/epigenetics_methylation.csv')
    methylation = e['epigenetics'].to_numpy()
    saliency_map(reconstructed_model, methylation)
