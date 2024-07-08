import os
import numpy as np
import tgt
from scipy.stats import mode


phoneme_list = ['aː˦˥', 'aː˦˨', 'aː˨ˀ˥', 'aː˨ˀ˦', 'aː˨˥', 'aː˨˦', 'aː˨˨', 'aː˨˨ˀ', 'aː˨˩', 'aː˨˩ˀ', 'aː˨˩˦', 'aː˨˩˨',
                'a˦˥', 'a˦˨', 'a˨ˀ˥', 'a˨ˀ˦', 'a˨˥', 'a˨˦', 'a˨˨', 'a˨˨ˀ', 'a˨˩', 'a˨˩ˀ', 'a˨˩˦', 'a˨˩˨',
                'c', 'eː˦˥', 'eː˦˨', 'eː˨ˀ˥', 'eː˨ˀ˦', 'eː˨˥', 'eː˨˦', 'eː˨˨', 'eː˨˨ˀ', 'eː˨˩', 'eː˨˩ˀ', 'eː˨˩˦', 'eː˨˩˨',
                'f', 'h',
                'iə˦˥', 'iə˦˨', 'iə˨ˀ˥', 'iə˨ˀ˦', 'iə˨˥', 'iə˨˦', 'iə˨˨', 'iə˨˨ˀ', 'iə˨˩', 'iə˨˩ˀ', 'iə˨˩˦', 'iə˨˩˨',
                'iː˦˥', 'iː˦˨', 'iː˨ˀ˥', 'iː˨ˀ˦', 'iː˨˥', 'iː˨˦', 'iː˨˨', 'iː˨˨ˀ', 'iː˨˩', 'iː˨˩ˀ', 'iː˨˩˦', 'iː˨˩˨',
                'j', 'k', 'kp',
                'l', 'm', 'n',
                'oː˦˥', 'oː˦˨', 'oː˨ˀ˥', 'oː˨ˀ˦', 'oː˨˥', 'oː˨˦', 'oː˨˨', 'oː˨˨ˀ', 'oː˨˩', 'oː˨˩ˀ', 'oː˨˩˦', 'oː˨˩˨',
                'p', 'r', 's', 't', 'tɕ', 'tʰ',
                'uə˦˥', 'uə˦˨', 'uə˨ˀ˥', 'uə˨ˀ˦', 'uə˨˥', 'uə˨˦', 'uə˨˨', 'uə˨˨ˀ', 'uə˨˩', 'uə˨˩ˀ', 'uə˨˩˦', 'uə˨˩˨',
                'uː˦˥', 'uː˦˨', 'uː˨ˀ˥', 'uː˨ˀ˦', 'uː˨˥', 'uː˨˦', 'uː˨˨', 'uː˨˨ˀ', 'uː˨˩', 'uː˨˩ˀ', 'uː˨˩˦', 'uː˨˩˨',
                'u˦˥', 'u˦˨', 'u˨ˀ˥', 'u˨ˀ˦', 'u˨˥', 'u˨˦', 'u˨˨', 'u˨˨ˀ', 'u˨˩', 'u˨˩ˀ', 'u˨˩˦', 'u˨˩˨',
                'v', 'w', 'x', 'z',
                'ŋ', 'ŋm', 'ɓ', 'ɔː˦˥', 'ɔː˦˨', 'ɔː˨ˀ˥', 'ɔː˨ˀ˦', 'ɔː˨˥', 'ɔː˨˦', 'ɔː˨˨', 'ɔː˨˨ˀ', 'ɔː˨˩', 'ɔː˨˩ˀ', 'ɔː˨˩˦', 'ɔː˨˩˨',
                'ɔ˦˥', 'ɔ˦˨', 'ɔ˨ˀ˥', 'ɔ˨ˀ˦', 'ɔ˨˥', 'ɔ˨˦', 'ɔ˨˨', 'ɔ˨˨ˀ', 'ɔ˨˩', 'ɔ˨˩ˀ', 'ɔ˨˩˦', 'ɔ˨˩˨',
                'ɗ', 'əː˦˥', 'əː˦˨', 'əː˨ˀ˥', 'əː˨ˀ˦', 'əː˨˥', 'əː˨˦', 'əː˨˨', 'əː˨˨ˀ', 'əː˨˩', 'əː˨˩ˀ', 'əː˨˩˦', 'əː˨˩˨',
                'ɛː˦˥', 'ɛː˦˨', 'ɛː˨ˀ˥', 'ɛː˨ˀ˦', 'ɛː˨˥', 'ɛː˨˦', 'ɛː˨˨', 'ɛː˨˨ˀ', 'ɛː˨˩', 'ɛː˨˩ˀ', 'ɛː˨˩˦', 'ɛː˨˩˨',
                'ɡ', 'ɣ', 'ɨə˦˥', 'ɨə˦˨', 'ɨə˨ˀ˥', 'ɨə˨ˀ˦', 'ɨə˨˥', 'ɨə˨˦', 'ɨə˨˨', 'ɨə˨˨ˀ', 'ɨə˨˩', 'ɨə˨˩ˀ', 'ɨə˨˩˦', 'ɨə˨˩˨',
                'ɨː˦˥', 'ɨː˦˨', 'ɨː˨ˀ˥', 'ɨː˨ˀ˦', 'ɨː˨˥', 'ɨː˨˦', 'ɨː˨˨', 'ɨː˨˨ˀ', 'ɨː˨˩', 'ɨː˨˩ˀ', 'ɨː˨˩˦', 'ɨː˨˩˨',
                'ɲ', 'ʈ', 'ʔ', 'ɨ˨˩˦', 'ə˨˩', 'ɨ˦˥', 'ɨ˨˩', 'k̚', 'spn', 'p̚', 'ɨ˨ˀ˥', 'ə˨˨', 'ɨ˨˩˨', 'ə˨˩˨', 'ɨ˨˨',
                'i˨˩˦', 'o˨˩˨', 'i˨˩',  'ə˦˥', 'ə˨˦', 'i˨˩˨', 'o˨˨', 't̚', 'i˨˨', 'ə˨˩˦', 'o˦˥', 'i˦˥', 'i˨˥',  'o˨˩', 'ə˨ˀ˦', 'i˦˨', 'ɨ˦˨',
                'o˨˦', 'ɨ˨˦', 'ə˨˩ˀ', 'i˨˦', 'o˨˩˦', 'i˨ˀ˦', 'o˨˩ˀ', 'ə˦˨', 'ə˨ˀ˥', 'i˨˩ˀ', 'ɨ˨˨ˀ', 'ə˨˨ˀ', 'ɨ˨˥',  'ə˨˥', 'i˨˨ˀ', 'ɨ˨ˀ˦',
                'ɨ˨˩ˀ', 'o˨˥',  'e˨˩ˀ', 'e˦˥', 'o˦˨',  'i˨ˀ˥', 'e˨˨', 'e˨˩', 'o˨ˀ˦']


phoneme_dict = dict()
for j, p in enumerate(phoneme_list):
    phoneme_dict[p] = j

data_dir = 'A'
mels_mode_dict = dict()
lens_dict = dict()
for p in phoneme_list:
    mels_mode_dict[p] = []
    lens_dict[p] = []
speakers = os.listdir(os.path.join(data_dir, 'mels'))
for s, speaker in enumerate(speakers):
    print('Speaker %d: %s' % (s + 1, speaker))
    textgrids = os.listdir(os.path.join(data_dir, 'textgrids', speaker))
    for textgrid in textgrids:
        t = tgt.io.read_textgrid(os.path.join(data_dir, 'textgrids', speaker, textgrid))
        # m = np.load(os.path.join(data_dir, 'mels', speaker, textgrid.replace('.TextGrid', '_mel.npy')))
        m = np.load(os.path.join(data_dir, 'mels', speaker, textgrid.replace('.TextGrid', '_mel.npy')))
        t = t.get_tier_by_name('phones')
        for i in range(len(t)):
            phoneme = t[i].text
            start_frame = int(t[i].start_time * 22050.0) // 256
            end_frame = int(t[i].end_time * 22050.0) // 256 + 1
            mels_mode_dict[phoneme] += [np.round(np.median(m[:, start_frame:end_frame], 1), 1)]
            lens_dict[phoneme] += [end_frame - start_frame]

mels_mode = dict()
lens = dict()
for p in phoneme_list:
    if len(mels_mode_dict[p]) > 0:
        mels_mode[p] = mode(np.asarray(mels_mode_dict[p]), 0).mode[0]
    lens[p] = np.mean(np.asarray(lens_dict[p]))
del mels_mode_dict
del lens_dict

for s, speaker in enumerate(speakers):
    print('Speaker %d: %s' % (s + 1, speaker))
    os.mkdir(os.path.join(data_dir, 'mels_mode', speaker))
    textgrids = os.listdir(os.path.join(data_dir, 'textgrids', speaker))
    for textgrid in textgrids:
        t = tgt.io.read_textgrid(os.path.join(data_dir, 'textgrids', speaker, textgrid))
        m = np.load(os.path.join(data_dir, 'mels', speaker, textgrid.replace('.TextGrid', '_mel.npy')))
        m_mode = np.copy(m)
        t = t.get_tier_by_name('phones')
        print(t)
        for i in range(len(t)):
            phoneme = t[i].text
            start_frame = int(t[i].start_time * 22050.0) // 256
            end_frame = int(t[i].end_time * 22050.0) // 256

            m_mode[:, start_frame:end_frame] = np.repeat(np.expand_dims(mels_mode[phoneme], 1), end_frame - start_frame, 1)

            #m_mode = m_mode[:, :27]
            

            print('bat dau')
            print("m_mode", m_mode)
            print("mels_mode[phoneme]",mels_mode[phoneme])
            print("start_frame", start_frame)
            print("end_frame", end_frame)

        np.save(os.path.join(data_dir, 'mels_mode', speaker, textgrid.replace('.TextGrid', '_avgmel.npy')), m_mode)

