############################################################
#	CIFAR10-ResNet benchmark (for VNN Comp 2022)		   #
#														   #
# Copyright (C) 2021  Shiqi Wang (sw3215@columbia.edu)	   #
# Copyright (C) 2021  Huan Zhang (huan@huan-zhang.com)	   #
# Copyright (C) 2021  Kaidi Xu (xu.kaid@northeastern.edu)  #
#														   #
# This program is licenced under the BSD 2-Clause License  #
############################################################

import os
import argparse
import csv
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from torch.utils.data import sampler

cifar100_mean = (0.5071, 0.4865, 0.4409)  # np.mean(train_set.train_data, axis=(0,1,2))/255
cifar100_std = (0.2673, 0.2564, 0.2761)  # np.std(train_set.train_data, axis=(0,1,2))/255
tinyimagenet_mean = (0.4802, 0.4481, 0.3975)
tinyimagenet_std = (0.2302, 0.2265, 0.2262)

def load_data(data_dir: str, dataset: str, num_imgs: int) -> tuple:
	"""
	Loads the cifar100 and tinyimagenet data.

	Args:
		data_dir:
			The directory to store the full CIFAR100 and TinyImageNet dataset.
		num_imgs:
			The number of images to extract from the test-set
		random:
			If true, random image indices are used, otherwise the first images
			are used.
	Returns:
		A tuple of tensors (images, labels).
	"""

	if not os.path.isdir(data_dir):
		os.mkdir(data_dir)
	if not os.path.isdir(data_dir + "/" + dataset):
		os.mkdir(data_dir + "/" + dataset)

	trns_norm = trans.ToTensor()
	if (dataset == "CIFAR100"):
		cifar100_test = dset.CIFAR100(data_dir, train=False, download=True, transform=trns_norm)
		loader_test = DataLoader(cifar100_test, batch_size=num_imgs)
	else:
		database_path = "./tiny-imagenet-200/val"
		if not os.path.isdir(database_path):
			os.system("bash ./tinyimagenet_download.sh")
		tinyimagenet_test = dset.ImageFolder(database_path,
									 transform=trans.Compose([
										trans.CenterCrop(56),
										trns_norm]))
		loader_test = DataLoader(tinyimagenet_test, batch_size=num_imgs)
	return next(iter(loader_test))


# noinspection PyShadowingNames
def create_input_bounds(img: torch.Tensor, eps: float,
						mean: tuple, std: tuple) -> torch.Tensor:
	"""
	Creates input bounds for the given image and epsilon.

	The lower bounds are calculated as img-eps clipped to [0, 1] and the upper bounds
	as img+eps clipped to [0, 1].

	Args:
		img:
			The image.
		eps:
		   The maximum accepted epsilon perturbation of each pixel.
		mean:
			The channel-wise means.
		std:
			The channel-wise standard deviation.
	Returns:
		A  img.shape x 2 tensor with the lower bounds in [..., 0] and upper bounds
		in [..., 1].
	"""

	mean = torch.tensor(mean, device=img.device).view(-1, 1, 1)
	std = torch.tensor(std, device=img.device).view(-1, 1, 1)

	bounds = torch.zeros((*img.shape, 2), dtype=torch.float32)
	bounds[..., 0] = (np.clip((img - eps), 0, 1) - mean) / std
	bounds[..., 1] = (np.clip((img + eps), 0, 1) - mean) / std
	# print(bounds[..., 0].abs().sum(), bounds[..., 1].abs().sum())

	return bounds.view(-1, 2)


# noinspection PyShadowingNames
def save_vnnlib(input_bounds: torch.Tensor, label: int, spec_path: str, total_output_class: int):
	"""
	Saves the classification property derived as vnn_lib format.

	Args:
		input_bounds:
			A Nx2 tensor with lower bounds in the first column and upper bounds
			in the second.
		label:
			The correct classification class.
		spec_path:
			The path used for saving the vnn-lib file.
		total_output_class:
			The total number of classification classes.
	"""

	with open(spec_path, "w") as f:

		f.write(f"; CIFAR100 property with label: {label}.\n")

		# Declare input variables.
		f.write("\n")
		for i in range(input_bounds.shape[0]):
			f.write(f"(declare-const X_{i} Real)\n")
		f.write("\n")

		# Declare output variables.
		f.write("\n")
		for i in range(total_output_class):
			f.write(f"(declare-const Y_{i} Real)\n")
		f.write("\n")

		# Define input constraints.
		f.write(f"; Input constraints:\n")
		for i in range(input_bounds.shape[0]):
			f.write(f"(assert (<= X_{i} {input_bounds[i, 1]}))\n")
			f.write(f"(assert (>= X_{i} {input_bounds[i, 0]}))\n")
			f.write("\n")
		f.write("\n")

		# Define output constraints.
		f.write(f"; Output constraints:\n")
		# orignal separate version:
		# for i in range(total_output_class):
		#	 if i != label:
		#		 f.write(f"(assert (>= Y_{label} Y_{i}))\n")
		# f.write("\n")

		# disjunction version:
		f.write("(assert (or\n")
		for i in range(total_output_class):
			if i != label:
				f.write(f"	(and (>= Y_{i} Y_{label}))\n")
		f.write("))\n")



def create_vnnlib(args, selected_list):

	assert os.path.exists(f"./onnx/{args.dataset}_resnet_{args.model}.onnx")
	instance_list = []
	epsilons = [eval(eps) for eps in args.epsilons.split(" ")]

	init_dir = f"./{args.mode}/".replace("generate", "generated").replace("_csv", "")
	if not os.path.isdir(init_dir):
		os.mkdir(init_dir)

	result_dir = init_dir

	print(result_dir)

	if not os.path.isdir(result_dir):
		os.mkdir(result_dir)

	if (args.dataset == "CIFAR100"):
		mu = torch.tensor(cifar100_mean).view(3, 1, 1)
		std = torch.tensor(cifar100_std).view(3, 1, 1)
	else:
		mu = torch.tensor(tinyimagenet_mean).view(3, 1, 1)
		std = torch.tensor(tinyimagenet_std).view(3, 1, 1)

	normalize = lambda X: (X - mu) / std

	images, labels = load_data(data_dir = "./", dataset = args.dataset, num_imgs=10000)
	np.random.seed(111)
	perm = np.random.permutation(images.shape[0])
	inv = np.argsort(perm)
	images, labels = images[perm], labels[perm]


	np.random.seed(args.seed)
	mask = np.zeros(len(selected_list))
	selected = np.random.permutation(len(selected_list))[:args.selected_instances]
	mask[selected] = True

	for eps in epsilons:
		cnt = 0
		for ii, i in enumerate(selected_list):
			# Load image and label.
			image, label = images[i], labels[i]
			image = image.unsqueeze(0)
			y = torch.tensor([label], dtype=torch.int64)

			print("scanned images: {}, selected: {}, label {}".format(i, cnt, label))

			if (mask[cnt] == 1):
				input_bounds = create_input_bounds(image, eps, mean = mu, std = std)
				spec_path = os.path.join(result_dir, f"{args.dataset}_resnet_{args.model}_prop_idx_{inv[i]}_sidx_{i}_eps_{eps:.4f}.vnnlib")
				if (args.dataset == "CIFAR100"):
					save_vnnlib(input_bounds, label, spec_path, total_output_class=100)
				else:
					save_vnnlib(input_bounds, label, spec_path, total_output_class=200)
		
				instance_list.append([f"{args.dataset}_resnet_{args.model}.onnx",
							  f"{args.dataset}_resnet_{args.model}_prop_idx_{inv[i]}_sidx_{i}_eps_0.0039.vnnlib", f"{args.timeout}"])
			cnt += 1

	assert os.path.exists(f"./generated_vnnlib/")
	
	with open(f'./cifar100_tinyimagenet_instances.csv', 'a') as f:
		write = csv.writer(f)
		write.writerows(instance_list)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('seed', type=int, default=0, help='random seed.') # seed for points selection	
	args = parser.parse_args()
	args.epsilons = "1/255"
	args.mode = "generate_vnnlib_csv"
	vnnlib_path = "generated_vnnlib"
	if not os.path.exists(vnnlib_path):
		os.makedirs(vnnlib_path)
	# Remove old files in the vnnlib folder.
	for vnnlib_file in os.scandir(vnnlib_path):
		os.remove(vnnlib_file.path)
	if os.path.exists("cifar100_tinyimagenet_instances.csv"):
		os.remove("cifar100_tinyimagenet_instances.csv")
	for dataset in ["CIFAR100", "TinyImageNet"]:
		args.dataset = dataset
		for model in ["small", "medium", "large", "super"]:
			if (dataset == "TinyImageNet" and model != "medium"): continue
			args.model = model
			selected_list = []
			ff = open(f"./selected_list/{args.dataset}_resnet_{args.model}_list.txt")
			for ss in ff: selected_list.append(int(ss))
			ff.close()
			print(len(selected_list))
			if (args.dataset == "CIFAR100"):
				if (args.model == "small"):
					args.selected_instances = 10
					args.timeout = 240
				elif (args.model == "super"):
					args.selected_instances = 16
					args.timeout = 300
				else:
					args.selected_instances = 24
					args.timeout = 200
			else:
				args.selected_instances = 24
				args.timeout = 200
			create_vnnlib(args, selected_list)


