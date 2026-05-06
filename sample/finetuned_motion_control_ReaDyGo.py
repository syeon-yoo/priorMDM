# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from diffusion.respace import SpacedDiffusion
from model.model_blending import ModelBlender
from utils.fixseed import fixseed
import os
import numpy as np
if not hasattr(np, "bool"):
    np.bool = np.bool_
    np.int = np.int_
    np.float = np.float_
    np.complex = np.complex128
    np.object = np.object_
    np.str = np.str_
    if not hasattr(np, "unicode"):
        np.unicode = np.str_
import torch
# from utils.parser_util import edit_inpainting_args
from utils.model_util import load_model_blending_and_diffusion
from utils import dist_util
from model.cfg_sampler import wrap_model
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml_utils import get_inpainting_mask
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from utils import parser_util
from argparse import ArgumentParser
from copy import deepcopy
from visualize.render_mesh_ReaDyGo import change_to_smpl_params

def edit_inpainting_args_readygo(model_path, guidance_param, output_dir):
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    parser_util.add_base_options(parser)
    parser_util.add_sampling_options(parser)
    parser_util.add_edit_inpainting_options(parser)
    
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    parser_util.add_data_options(parser)
    parser_util.add_model_options(parser)
    parser_util.add_diffusion_options(parser)
    
    base_argv = [
        "--model_path", model_path,
        "--output_dir", output_dir or "",
        # "--guidance_param", str(guidance_param),
        "--text_condition", "a person is naturally walking forward along the trajectory",
        "--num_samples", "1",
    ]
    
    args, _ = parser.parse_known_args(base_argv)
    model_paths = args.model_path.split(',')
    args_list = []
    for i in range(len(model_paths)):
        new_args = deepcopy(args)
        new_args.model_path = model_paths[i]
        args_list.append(parser_util.load_from_model(new_args, parser, task='inpainting'))
    
    if args.inpainting_mask == '':
        inpainting_mask = ','.join([args.inpainting_mask for args in args_list])
        for args in args_list:
            args.inpainting_mask = inpainting_mask
    print(f'Using inpainting mask: {args_list[0].inpainting_mask}')
    return args_list

def human_2D_path_to_SMPL_seq(model_path, guidance_param, output_dir, traj, duration, hugs_smpl_betas_path=None):
    args_list = edit_inpainting_args_readygo(model_path, guidance_param, output_dir)
    args = args_list[0]
    fixseed(args.seed)
    
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'edit_{}_{}_{}_seed{}'.format(name, niter, args.inpainting_mask, args.seed))
        if args.text_condition != '':
            out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')

    print('Loading dataset...')
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              load_mode='train',
                            #   size=args.num_samples,
                              )  # in train mode, you get both text and motion.
    # data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    DiffusionClass = InpaintingGaussianDiffusion if args_list[0].filter_noise else SpacedDiffusion
    model, diffusion = load_model_blending_and_diffusion(args_list, data, dist_util.dev(), DiffusionClass=DiffusionClass)

    iterator = iter(data)
    # create my own input_motions and model_kwargs to run the model on my own trajectory (Let's make temp trajectory and verify the results)
    input_motions, model_kwargs = next(iterator)
    #######################################################################
    ######################## input preprocessing ##########################
    ###### fps=20, max_frames=196, args.batch_size=args.num_samples #######    
    
    # Compute (Δx, Δy, Δyaw)
    def world_to_local_vel(x_world, y_world):
        # Our world coordinate system: xy-plane (horizontal), z-axis (up). x-right, y-forward
        # The translation in world xy-plane per frame (m/frame)
        dx = np.diff(x_world, prepend=x_world[0])
        dy = np.diff(y_world, prepend=y_world[0])
        dx[0] = dx[1]
        dy[0] = dy[1]
        dx = -dx
        # The root horizontal angle in world xy-plane to be tangent to the trajectory. i.e., yaw=atan2(Δy, Δx), Δyaw(rad/frame)
        yaw = np.unwrap(np.arctan2(dy, dx))
        # yaw = np.convolve(yaw, np.ones(5) / 5.0, mode='same')
        dyaw = np.diff(yaw, prepend=yaw[0])
        dyaw[0] = dyaw[1]
        
        # PriorMDM coordinate system: xz-plane (horizontal), y_axis (up)
        # 263-dim HumanML3D vector requires world root angular velocity and local linear velocities
        # v_fwd & v_side: rel linear vel, omega: y-axis world angular vel
        c, s = np.cos(yaw), np.sin(yaw)
        v_fwd = c*dx + s*dy                 # sqrt(dx^2+dz^2). Rel linear vel. (m/frame)
        v_side = -s*dx + c*dy                 # Zero if tangential motion. Rel linear vel. (m/frame)
        omega = dyaw / 2

        return v_fwd, v_side, omega, yaw  # (T,),(T,),(T,),(T,)

    def canonical_to_world_xz_motion(motion, heading, traj_xy):
        """Rotate local xz motion back to the world_xz 2D trajectory."""
        world_xz_motion = np.empty_like(motion)
        T = motion.shape[-1]
        if args.dataset == 'kit':
            r_hip, l_hip = 11, 16
        else:
            r_hip, l_hip = 2, 1
        for t in range(T):
            hip_vec = motion[l_hip, :, t] - motion[r_hip, :, t]
            current_yaw = np.arctan2(hip_vec[2], hip_vec[0])
            theta = (np.pi / 2.0) - heading[t] + current_yaw
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ])
            rotated = (rot @ motion[:, :, t].T).T
            target_root = np.array([-traj_xy[t, 0], rotated[0, 1], traj_xy[t, 1]])
            world_xz_motion[:, :, t] = target_root + (rotated - rotated[0])
        return world_xz_motion
    
    # === make input_motions (B,263,1,196): only fill 0~2 channels before diffusion process ===
    device = dist_util.dev()
    input_motions = torch.zeros(args.batch_size, 263, 1, max_frames, device=device, dtype=torch.float32)
    heading_cache = []
    traj_cache = []
    lengths_list = []
    for i in range(args.num_samples):
        assert args.num_samples == 1
        Ti = int(min(max_frames, round(duration * fps)))  # human scenario length
        x_world, y_world = np.array(traj).T
        v_fwd, v_side, omega, heading = world_to_local_vel(x_world, y_world)  # (m/frame), (rad/frame)
        # channel 0: world root_rot_velocity, 1~2: relative root_linear_velocity
        omega_norm = (omega - data.dataset.mean[0]) / (data.dataset.std[0])
        v_side_norm = (v_side - data.dataset.mean[1]) / (data.dataset.std[1])
        v_fwd_norm = (v_fwd - data.dataset.mean[2]) / (data.dataset.std[2])
        input_motions[i, 0, 0, :Ti] = torch.from_numpy(omega_norm[:Ti]).to(device).float()
        input_motions[i, 1, 0, :Ti] = torch.from_numpy(v_side_norm[:Ti]).to(device).float()
        input_motions[i, 2, 0, :Ti] = torch.from_numpy(v_fwd_norm[:Ti]).to(device).float()

        lengths_list.append(Ti)
        traj_xy = np.stack([x_world, y_world], axis=-1)
        if Ti < max_frames:
            assert Ti > 0
            heading_pad = np.full(max_frames - Ti, heading[Ti - 1])
            traj_pad = np.repeat(traj_xy[Ti - 1:Ti], max_frames - Ti, axis=0)
            heading_full = np.concatenate([heading[:Ti], heading_pad], axis=0)
            traj_full = np.concatenate([traj_xy[:Ti], traj_pad], axis=0)
        else:
            heading_full = heading[:max_frames]
            traj_full = traj_xy[:max_frames]
        heading_cache.append(heading_full)
        traj_cache.append(traj_full)
    
    lengths = torch.tensor(lengths_list, dtype=torch.long, device=device)
    model_kwargs['y']['lengths'] = lengths
    #######################################################################
    #######################################################################
    
    input_motions = input_motions.to(dist_util.dev())
    if args.text_condition != '':
        texts = [args.text_condition] * args.num_samples
        model_kwargs['y']['text'] = texts

    # add inpainting mask according to args
    assert max_frames == input_motions.shape[-1]
    gt_frames_per_sample = {}
    model_kwargs['y']['inpainted_motion'] = input_motions
    model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, input_motions.shape)).float().to(dist_util.dev())

    all_motions = []
    all_lengths = []
    all_text = []

    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=input_motions,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )


        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        all_text += model_kwargs['y']['text']
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    world_motions = []
    for sample_idx in range(all_motions.shape[0]):
        ref_idx = sample_idx % args.num_samples
        world_motions.append(
            canonical_to_world_xz_motion(all_motions[sample_idx], heading_cache[ref_idx], traj_cache[ref_idx])
        )
    all_motions = np.stack(world_motions, axis=0)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'priorMDM_results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        input_motions = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
        input_motions = recover_from_ric(input_motions, n_joints)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
        for sample_i in range(args.num_samples):
            input_motions[sample_i] = canonical_to_world_xz_motion(
                input_motions[sample_i], heading_cache[sample_i], traj_cache[sample_i]
            )


    for sample_i in range(args.num_samples):
        rep_files = []
        if args.show_input:
            caption = 'Input Motion'
            length = model_kwargs['y']['lengths'][sample_i]
            motion = input_motions[sample_i].transpose(2, 0, 1)[:length]
            save_file = 'input_motion{:02d}.mp4'.format(sample_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({sample_i}) "{caption}" | -> {save_file}]')
            plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                        dataset=args.dataset, fps=fps, vis_mode='gt',
                        gt_frames=gt_frames_per_sample.get(sample_i, []))
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + sample_i]
            if args.guidance_param == 0:
                caption = 'Edit [{}] unconditioned'.format(args.inpainting_mask)
            else:
                caption = 'Edit [{}]: {}'.format(args.inpainting_mask, caption)
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {animation_save_path}]')
            plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                           dataset=args.dataset, fps=fps, vis_mode=args.inpainting_mask,
                           gt_frames=gt_frames_per_sample.get(sample_i, []), painting_features=args.inpainting_mask.split(','))
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        if args.num_repetitions > 1:
            all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
            ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
            hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions + (1 if args.show_input else 0)} '
            ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
            os.system(ffmpeg_rep_cmd)
            print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    # change xyz positions (npy_path) to smpl params using SMPLify and save
    change_to_smpl_params(animation_save_path, hugs_smpl_betas_path=hugs_smpl_betas_path)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')

