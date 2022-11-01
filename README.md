# About
In this repository we provide the prototype implementation of the MCL GLOSM/ELOFF filter proposed in our paper in [1].

Even though this repository implements the code for reproducing the results presented in [1], we also provide, for the convenience of the readers, the links with the results obtained in our experiments in the links below.
* [Videos](https://drive.google.com/drive/folders/1Hxd4Wx8YmKxxPYoSmxUOW6Kdlk4OxrRT?usp=share_link)
* [Trajectory output files](https://drive.google.com/drive/folders/1QBx2z7TTumm5LhN--c_Cp71DJZIXYKDh?usp=share_link)
* [Detailed metrics' spreadsheet](https://docs.google.com/spreadsheets/d/1aWFTAXCy8KD8N8RbJlWHImM4x3Cwvl-UQu2UBfMs6wY/edit?usp=sharing)
* [Digital surface maps](https://drive.google.com/drive/folders/1GMxU_V90U7bW9dqKdiR6_qPZxR5k8N-p?usp=share_link)
* [Point Cloud PLY files](https://drive.google.com/drive/folders/1bFP7Hl6C91-NOYAgEIBzKAg4-jlgxgQk?usp=share_link).

Related repositories:
* [SLOPY](https://github.com/cabraile/SLOPY): Point-to-plane odometry implemented for validating the approach;
* [DF-VO](https://github.com/Huangying-Zhan/DF-VO): Odometry source used for comparing SLOPY for the Kitti demo;
* [3D Kitti Mapper](https://github.com/cabraile/3D-Kitti-Mapper): Maps the Kitti trajectories and export them as PLY. Also provide the tools for exporting the mapped trajectories to digital surface map raster files.

If having trouble for associating the `date`/`drive` to their respective sequences, we provide the mappings under `scripts/demos/kitti/sequence_utils.py`.

# Setup
For running the demos, we suggest the user to prepare an Anaconda environment using the `environment.yml` file.

# Kitti demo
For evaluating our method, we implemented a script for running the Kitti sequences under `scripts/demos/kitti/demo_kitti.py`.

## Setting the configuration file
In the configuration file under `cfg/demo.yaml`, you will find the parameters for the particle filter, SLOPY, GLOSM, ELOFF and the output video demo. Even though not necessary to change (our experiments were executed using those), you can adjust the parameters as you would like there. 

The parameters are detailed in the file.

## Running the pipeline

Run the command for executing the SLOPY+GLOSM+ELOFF pipeline with the required arguments:
```bash
python3 scripts/demos/kitti/demo_kitti.py \
    --dataset ${DATASET_DIR} \
    --sequence_id ${SEQUENCE} \
    --dsm_path ${DSM_PATH}
```

The required arguments are:
* `--dataset`, the absolute path to the Kitti dataset extracted in the date-drive format as expected by the [PyKitti toolkit](https://github.com/utiasSTARS/pykitti);
* `--sequence_id`, the id of the sequence (`00`, `01`,..., `10`) to be tested;
* `--dsm_path`, the absolute path to the DSM of the sequence.

Other arguments that are optional are detailed if you run `python3 scripts/demos/kitti/demo_kitti.py -h`.

## Citation
**To be included.**