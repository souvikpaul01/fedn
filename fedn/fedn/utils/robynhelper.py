import os
import tempfile
import numpy as np
import collections
import tempfile

from .helpers import HelperBase


class RobynHelper(HelperBase):
    """ FEDn helper class for keras.Sequential. """

    def increment_average(self, weights1, weights2, n):
        """ Update an incremental average. """

        new_features = list(set(weights2['features']) - set(weights1['features']))
        missing_features = list(set(weights1['features']) - set(weights2['features']))

        # update weights1
        weights1['features'] += new_features
        weights1['values'] = np.concatenate(
            (weights1['values'], np.zeros((len(new_features), weights1['values'].shape[1]))), 0)
        sort1 = np.argsort(weights1['features'])
        weights1['features'] = list(np.array(weights1['features'])[sort1])
        weights1['values'] = weights1['values'][sort1]

        # update weights2
        weights2['features'] += missing_features
        weights2['values'] = np.concatenate(
            (weights2['values'], np.zeros((len(missing_features), weights2['values'].shape[1]))), 0)
        sort2 = np.argsort(weights2['features'])
        weights2['features'] = list(np.array(weights2['features'])[sort2])
        weights2['values'] = weights2['values'][sort2]

        # updated weights
        weights3 = {'features': weights1['features'],
                    'values': np.concatenate((weights1['values'], weights2['values']), 1)}
        return weights3

    def set_weights(self, weights_, weights):
        """
        :param weights_:
        :param weights:
        """
        weights_ = weights

    def get_weights(self, weights):
        """
        :param weights:
        :return:
        """
        return weights

    def get_tmp_path(self):
        """ Return a temporary output path compatible with save_model, load_model. """
        fd, path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)
        return path

    def save_model(self, weights, path=None):
        """
        :param weights:
        :param path:
        :return:
        """
        print("mattias robyn saving_model")
        if not path:
            path = self.get_tmp_path()

        np.savez_compressed(path, **weights)

        return path

    def load_model(self, path="weights.npz"):
        """
        :param path:
        :return:
        """
        b = np.load(path)
        weights_np = {}
        for i in b.files:
            weights_np[i] = b[i]
        return weights_np

    def get_tmp_path(self):
        """ Return a temporary output path compatible with save_model, load_model. """
        fd, path = tempfile.mkstemp(suffix='.npz')
        os.close(fd)
        return path



    def load_model_from_BytesIO(self, model_bytesio):
        """ Load a model from a BytesIO object. """
        path = self.get_tmp_path()
        with open(path, 'wb') as fh:
            fh.write(model_bytesio)
            fh.flush()
        model = self.load_model(path)
        os.unlink(path)
        return model

    def serialize_model_to_BytesIO(self, model):
        """

        :param model:
        :return:
        """
        outfile_name = self.save_model(model)

        from io import BytesIO
        a = BytesIO()
        a.seek(0, 0)
        with open(outfile_name, 'rb') as f:
            a.write(f.read())
        os.unlink(outfile_name)
        return a
