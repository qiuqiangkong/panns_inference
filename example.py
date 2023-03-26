import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels


def print_audio_tagging_result(clipwise_output):
    """Visualization of audio tagging result.

    Args:
      clipwise_output: (classes_num,)
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))


def plot_sound_event_detection_result(framewise_output):
    """Visualization of sound event detection result. 

    Args:
      framewise_output: (time_steps, classes_num)
    """
    out_fig_path = 'results/sed_result.png'
    os.makedirs(os.path.dirname(out_fig_path), exist_ok=True)

    classwise_output = np.max(framewise_output, axis=0) # (classes_num,)

    idxes = np.argsort(classwise_output)[::-1]
    idxes = idxes[0:5]

    ix_to_lb = {i : label for i, label in enumerate(labels)}
    lines = []
    for idx in idxes:
        line, = plt.plot(framewise_output[:, idx], label=ix_to_lb[idx])
        lines.append(line)

    plt.legend(handles=lines)
    plt.xlabel('Frames')
    plt.ylabel('Probability')
    plt.ylim(0, 1.)
    plt.savefig(out_fig_path)
    print('Save fig to {}'.format(out_fig_path))


if __name__ == '__main__':
    """Example of using panns_inferece for audio tagging and sound evetn detection.
    """
    device = 'cpu' # 'cuda' | 'cpu'
    audio_path = 'resources/R9_ZSCveAHg_7s.wav'
    (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
    audio = audio[None, :]  # (batch_size, segment_samples)

    print('------ Audio tagging ------')
    at = AudioTagging(checkpoint_path=None, device=device)
    (clipwise_output, embedding) = at.inference(audio)
    """clipwise_output: (batch_size, classes_num), embedding: (batch_size, embedding_size)"""

    print_audio_tagging_result(clipwise_output[0])

    print('------ Sound event detection ------')
    sed = SoundEventDetection(
        checkpoint_path=None, 
        device=device, 
        interpolate_mode='nearest', # 'nearest'
    )
    framewise_output = sed.inference(audio)
    """(batch_size, time_steps, classes_num)"""

    plot_sound_event_detection_result(framewise_output[0])