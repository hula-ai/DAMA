#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert DAMA Pre-Traind Model to DEiT')
    parser.add_argument('--input', default='', type=str, metavar='PATH', required=True,
                        help='path to DAMA pre-trained checkpoint')
    parser.add_argument('--output', default='', type=str, metavar='PATH', required=True,
                        help='path to output checkpoint in DEiT format')
    args = parser.parse_args()
    print(args)

    # load input
    checkpoint = torch.load(args.input, map_location="cpu")
    state_dict = checkpoint['model']

    for k in list(state_dict.keys()):
        # retain only base_encoder (student) up to 1st layer of decoder layer
        if k.startswith('base_encoder') and 'decoder' not in k and 'mask' not in k and 'momentum_encoder' not in k:
            # remove prefix
            state_dict[k[len("base_encoder."):]] = state_dict[k]
        del state_dict[k]
    print('Loading base encoder model.')


    # make output directory if necessary
    output_dir = os.path.dirname(args.output)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # save to output
    torch.save({'model': state_dict}, args.output)