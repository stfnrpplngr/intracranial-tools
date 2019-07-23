"""
This file is copyright under the latest version of the EUPL.
Please see LICENSE file for your rights under this license.

Author: Stefan Repplinger
Mail: stefan.repplinger@ovgu.de
URL: www.neuropsychologie.ovgu.de

(c) 2019 stfnrpplngr
"""

import numpy as np


def read_inomed_trg(DataChan, data_info):
    """Convert trigger channel voltages into integers.

    Parameters
    ----------
    DataChan : array
        Channel data coming from 'load_EMG_Data.m'.
    data_info : dict
        Dictionary containing header read-out.

    Returns
    -------
    chan_trg : array
        Array containing trigger code for each sample.

    """
    # duration of one sample
    sample_dur = 1 / data_info['SampFreq'].item()
    # array with sample timings
    sample_times = np.arange(0, len(DataChan) * sample_dur, sample_dur)
    if len(sample_times) > len(DataChan):
        print('Mismatch in time array generation.')
        sample_times = sample_times[:-1]

    # factor to convert trigger voltage values into integers
    trg_factor = (2 ** -149) * 4

    # adding a column with sample times to DataChan
    DataChan_times = np.append(DataChan, np.expand_dims(sample_times, 1), 1)
    # select lines containing trigger signal > 0
    DataChan_times = DataChan_times[:, np.sum(DataChan_times, 0) > 0]
    DataChan_times = DataChan_times[np.sum(DataChan_times[:, :-1], 1) > 0, :]
    if DataChan_times.size == 0:
        print('No trigger signal found')
    # converting trigger signals into integers
    DataChan_times[:, :-1] = DataChan_times[:, :-1] / trg_factor

    # calculating the time difference between consecutive samples
    trg_timediff = np.insert(
        np.diff(DataChan_times[:, -1]), 0, 1)[:, np.newaxis]
    DataChan_times = np.append(DataChan_times, trg_timediff, 1)

    # convert trigger integers into trigger codes
    for idx in range(1, 3):
        DataChan_times[DataChan_times[:, idx].nonzero(), idx] += 7*idx

    # sample_dist -> minimal number of samples between data packages
    sample_dist = 40
    trg_paketstart = np.where(
        DataChan_times[:, -1] > sample_dur * sample_dist)[0]
    trg_paketstart = np.append(trg_paketstart, len(DataChan_times))

    # timestamps of triggers are determined according to package onsets
    trg_timestamps = DataChan_times[trg_paketstart[:-1], 3].transpose()

    # trigger codes are read out
    trg_codes = np.array(list())
    for idx1, idx2 in zip(trg_paketstart[:-1], trg_paketstart[1:]):
        trg_codes = np.append(trg_codes, np.sum(DataChan_times[idx1:idx2, :3]))
    trg_codes[trg_codes == 37], trg_codes[trg_codes == 38] = 254, 255

    # determining indices of trigger onsets
    trg_indices = np.isin(sample_times, trg_timestamps).nonzero()[0]
    chan_trg = np.zeros([1, len(sample_times)])
    chan_trg[0, trg_indices] = trg_codes

    return chan_trg
