from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import os

import librosa
import numpy as np
import tensorflow as tf

import time

from wavenet import WaveNetModel, mu_law_decode, mu_law_encode, audio_reader

SAMPLES = 16000
TEMPERATURE = 1.0
LOGDIR = './logdir'
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = None
SILENCE_THRESHOLD = 0.1

os.environ["CUDA_VISIBLE_DEVICES"]="1" #Use Only GPU 3

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError(
                    'Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument(
        'checkpoint', type=str, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--samples',
        type=int,
        default=SAMPLES,
        help='How many waveform samples to generate')
    parser.add_argument(
        '--temperature',
        type=_ensure_positive_float,
        default=TEMPERATURE,
        help='Sampling temperature')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='JSON file with the network parameters')
    parser.add_argument(
        '--wav_out_path',
        type=str,
        default=None,
        help='Path to output wav file')
    parser.add_argument(
        '--save_every',
        type=int,
        default=SAVE_EVERY,
        help='How many samples before saving in-progress wav')
    parser.add_argument(
        '--fast_generation',
        type=_str_to_bool,
        default=True,
        help='Use fast generation')
    parser.add_argument(
        '--wav_seed',
        type=str,
        default=None,
        help='The wav file to start generation from')
    parser.add_argument(
        '--gc_channels',
        type=int,
        default=None,
        help='Number of global condition embedding channels. Omit if no '
             'global conditioning.')
    parser.add_argument(
        '--gc_cardinality',
        type=int,
        default=None,
        help='Number of categories upon which we globally condition.')
    parser.add_argument(
        '--lc_channels',
        type=bool,
        default=None,
        help='Number of local condition embedding channels. Omit if no '
             'local conditioning.')
    parser.add_argument(
        '--lc_cardinality',
        type=int,
        default=None,
        help='Number of categories upon which we local condition.')
    parser.add_argument(
        '--gc_id',
        type=int,
        default=None,
        help='ID of category to generate, if globally conditioned.')
    #parser.add_argument(
    #    '--lc_id',
    #    type=int,
    #    default=None,
    #    help='ID of category to generate, if locally conditioned.')
    parser.add_argument(
        '--labels',
        type=str,
        default=None,
        help='Name of the file that contains labels for each samples.')

    arguments = parser.parse_args()
    if arguments.gc_channels is not None:
        if arguments.gc_cardinality is None:
            raise ValueError("Globally conditioning but gc_cardinality not "
                             "specified. Use --gc_cardinality=377 for full "
                             "VCTK corpus.")

        if arguments.gc_id is None:
            raise ValueError("Globally conditioning, but global condition was "
                              "not specified. Use --gc_id to specify global "
                              "condition.")

    if arguments.lc_channels is not None:
        if arguments.lc_cardinality is None:
            raise ValueError("Locally conditioning but lc_cardinality not "
                             "specified." )

        #if arguments.lc_id is None:
        #    raise ValueError("Locally conditioning, but local condition was "
        #                      "not specified. Use --lc_id to specify global "
        #                      "condition.")

    return arguments


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


def create_seed(filename,
                sample_rate,
                quantization_channels,
                window_size,
                silence_threshold=SILENCE_THRESHOLD):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio_reader.trim_silence(audio, silence_threshold)

    quantized = mu_law_encode(audio, quantization_channels)
    cut_index = tf.cond(tf.size(quantized) < tf.constant(window_size),
                        lambda: tf.size(quantized),
                        lambda: tf.constant(window_size))

    return quantized[:cut_index]

#aleix
def read_sample_label(labelsFileName):
    with open(labelsFileName, 'r') as myfile:
        labels_sample = myfile.read().replace('\n', '')

    return(labels_sample)
#aleix


def main():
    args = get_arguments()
    if args.lc_channels is not None:
    	args.lc_channels = args.lc_cardinality*3
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    logdir = os.path.join(args.logdir, 'generate', started_datestring)
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session()

    net = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        quantization_channels=wavenet_params['quantization_channels'],
        skip_channels=wavenet_params['skip_channels'],
        use_biases=wavenet_params['use_biases'],
        scalar_input=wavenet_params['scalar_input'],
        initial_filter_width=wavenet_params['initial_filter_width'],
        global_condition_channels=args.gc_channels,
        global_condition_cardinality=args.gc_cardinality,
        local_condition_channels = args.lc_channels,
        local_condition_cardinality = args.lc_cardinality)


    samples = tf.placeholder(tf.int32)
    if args.labels is not None:
        sample_labels = tf.placeholder(tf.int32) #aleix
    else:
        sample_labels = None


    if args.fast_generation:
        #next_sample = net.predict_proba_incremental(samples, args.gc_id, args.lc_id)
        next_sample = net.predict_proba_incremental(samples, args.gc_id, sample_labels)
    else:
        #next_sample = net.predict_proba(samples, args.gc_id, args.lc_id)
        next_sample = net.predict_proba_incremental(samples, args.gc_id, sample_labels)

    if args.fast_generation:
        sess.run(tf.global_variables_initializer())
        sess.run(net.init_ops)

    variables_to_restore = {
        var.name[:-2]: var for var in tf.global_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    start_time = time.time()
    saver.restore(sess, args.checkpoint)

    decode = mu_law_decode(samples, wavenet_params['quantization_channels'])

    quantization_channels = wavenet_params['quantization_channels']
    if args.wav_seed:
        seed = create_seed(args.wav_seed,
                           wavenet_params['sample_rate'],
                           quantization_channels,
                           net.receptive_field)
        waveform = sess.run(seed).tolist()
    else:
        # Silence with a single random sample at the end.
        waveform = [quantization_channels / 2] * (net.receptive_field - 1)
        waveform.append(np.random.randint(quantization_channels))

    if args.fast_generation and args.wav_seed:
        # When using the incremental generation, we need to
        # feed in all priming samples one by one before starting the
        # actual generation.
        # TODO This could be done much more efficiently by passing the waveform
        # to the incremental generator as an optional argument, which would be
        # used to fill the queues initially.
        outputs = [next_sample]
        outputs.extend(net.push_ops)

        print('Priming generation...')
        for i, x in enumerate(waveform[-net.receptive_field: -1]):
            if i % 100 == 0:
                print('Priming sample {}'.format(i))
            sess.run(outputs, feed_dict={samples: x})
        print('Done.')

    last_sample_timestamp = datetime.now()

    # aleix
    # sample_labels_list = [0]*8000+[1]*8000
    if args.labels is not None:
        labelsFileName = (args.labels)
        sample_labels_list = read_sample_label(labelsFileName)
        sample_labels_list = np.fromstring(sample_labels_list, dtype=int, sep=',').reshape(-1, 1)

        # Convert to oneHot. Cannot be a tensor, so use numpy instead
        # Add previous, current, next
        for i in range(len(sample_labels_list)):
            if i == 0:
                sample_labels_list_prev = np.array([0])
                sample_labels_list_next = np.array(sample_labels_list[i + 1])
            elif i == len(sample_labels_list) - 1:
                sample_labels_list_prev = np.append(sample_labels_list_prev, sample_labels_list[i - 1])
                sample_labels_list_next = np.append(sample_labels_list_next, sample_labels_list[i])
            else:
                sample_labels_list_prev = np.append(sample_labels_list_prev, sample_labels_list[i - 1])
                sample_labels_list_next = np.append(sample_labels_list_next, sample_labels_list[i + 1])

        sample_labels_list = sample_labels_list.reshape(1, -1)
        sample_labels_list_prev = sample_labels_list_prev.reshape(1, -1)
        sample_labels_list_next = sample_labels_list_next.reshape(1, -1)

        sample_labels_list = np.eye(int(args.lc_channels / 3))[sample_labels_list][0]
        sample_labels_list_prev = np.eye(int(args.lc_channels / 3))[sample_labels_list_prev][0]
        sample_labels_list_next = np.eye(int(args.lc_channels / 3))[sample_labels_list_next][0]

        sample_labels_list = np.append(sample_labels_list_prev, sample_labels_list, axis=1)
        sample_labels_list = np.append(sample_labels_list, sample_labels_list_next, axis=1)

        #sample_labels_list =sample_labels_list.reshape(1, -1)
        #sample_labels_list = np.eye(args.lc_channels)[sample_labels_list][0]
    else:
        sample_labels_list = None

    for step in range(args.samples):
        if args.fast_generation:
            outputs = [next_sample]
            outputs.extend(net.push_ops)
            window = waveform[-1]
            if sample_labels_list is not None:
                label_window = sample_labels_list[step]
            else:
                label_window = None
        else:
            if len(waveform) > net.receptive_field:
                window = waveform[-net.receptive_field:]
            else:
                window = waveform
            outputs = [next_sample]

        # Run the WaveNet to predict the next sample.
        #prediction = sess.run(outputs, feed_dict={samples: window})[0]
        if args.labels is not None:
            prediction = sess.run(outputs, feed_dict={samples: window, sample_labels: label_window})[0]
        else:
            prediction = sess.run(outputs, feed_dict={samples: window})[0]



        # Scale prediction distribution using temperature.
        np.seterr(divide='ignore')
        scaled_prediction = np.log(prediction) / args.temperature
        scaled_prediction = (scaled_prediction -
                             np.logaddexp.reduce(scaled_prediction))
        scaled_prediction = np.exp(scaled_prediction)
        np.seterr(divide='warn')

        # Prediction distribution at temperature=1.0 should be unchanged after
        # scaling.
        if args.temperature == 1.0:
            np.testing.assert_allclose(
                    prediction, scaled_prediction, atol=1e-5,
                    err_msg='Prediction scaling at temperature=1.0 '
                            'is not working as intended.')

        sample = np.random.choice(
            np.arange(quantization_channels), p=scaled_prediction)
        waveform.append(sample)

        # Show progress only once per second.
        current_sample_timestamp = datetime.now()
        time_since_print = current_sample_timestamp - last_sample_timestamp
        if time_since_print.total_seconds() > 1.:
            print('Sample {:3<d}/{:3<d}'.format(step + 1, args.samples),
                  end='\r')
            last_sample_timestamp = current_sample_timestamp

        # If we have partial writing, save the result so far.
        if (args.wav_out_path and args.save_every and
                (step + 1) % args.save_every == 0):
            out = sess.run(decode, feed_dict={samples: waveform})
            write_wav(out, wavenet_params['sample_rate'], args.wav_out_path)

    # Introduce a newline to clear the carriage return from the progress.
    print()

    # Save the result as an audio summary.
    datestring = str(datetime.now()).replace(' ', 'T')
    writer = tf.summary.FileWriter(logdir)
    tf.summary.audio('generated', decode, wavenet_params['sample_rate'])
    summaries = tf.summary.merge_all()
    summary_out = sess.run(summaries,
                           feed_dict={samples: np.reshape(waveform, [-1, 1])})
    writer.add_summary(summary_out)

    # Save the result as a wav file.
    if args.wav_out_path:
        out = sess.run(decode, feed_dict={samples: waveform})
        write_wav(out, wavenet_params['sample_rate'], args.wav_out_path)

    print('Finished generating. The result can be viewed in TensorBoard.')
    print()
    duration = time.time() - start_time
    print('Generation time: %s sec' % duration)


if __name__ == '__main__':
    main()
