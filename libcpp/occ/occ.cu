#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define MAX_D 1446 // 700 + 700 + 45 + 1

template <typename scalar_t>
__global__ void pcld2occ3d_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> occupancy) {

    // batch index
    const auto b = blockIdx.y;

    // point index
    const auto n = blockIdx.x * blockDim.x + threadIdx.x;

    // num of points
    const auto N = points.size(1);

    // we allocated more threads than num_rays
    if (n < N) {
        // grid shape
        const int vzsize = occupancy.size(1);
        const int vysize = occupancy.size(2);
        const int vxsize = occupancy.size(3);

        // end point
        const int vx = int(points[b][n][0]);
        const int vy = int(points[b][n][1]);
        const int vz = int(points[b][n][2]);

        //
        if (0 <= vx && vx < vxsize &&
            0 <= vy && vy < vysize &&
            0 <= vz && vz < vzsize) {
            occupancy[b][vz][vy][vx] = 1;
        }
    }
}

/* pcld2occ3d_cuda
 * input shape
 *   points   : B x N x 3
 *   grid     : 4
 * output shape
 *   occupancy: B x H x L x W
 */
torch::Tensor pcld2occ3d_cuda(
    torch::Tensor points,
    const std::vector<int> grid) {

    const auto B = points.size(0); // batch size
    const auto N = points.size(1); // number of points

    const auto T = grid[0]; // time span (not used currently)
    const auto H = grid[1]; // grid height
    const auto L = grid[2]; // grid length
    const auto W = grid[3]; // grid width

    const auto dtype = points.dtype();
    const auto device = points.device();
    const auto options = torch::TensorOptions().dtype(dtype).device(device).requires_grad(false);
    auto occupancy = torch::zeros({B, H, L, W}, options);

    const int threads = 1024;
    const dim3 blocks((N + threads - 1) / threads, B);

    // initialize occupancy such that every voxel with one or more points is occupied
    AT_DISPATCH_FLOATING_TYPES(points.type(), "pcld2occ3d_cuda", ([&] {
                pcld2occ3d_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    occupancy.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
            }));

    // synchronize
    cudaDeviceSynchronize();

    return occupancy;
}

/* render demo */

template <typename scalar_t>
__global__ void render_demo_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> occ,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> pixel,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> rotation,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> translation,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> scale,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dist,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> depth) {
        const auto h = blockIdx.x * blockDim.x + threadIdx.x;
        const auto w = blockIdx.y * blockDim.y + threadIdx.y;

        const auto H = pixel.size(0);
        const auto W = pixel.size(1);

        const auto offset_x = scale[0];
        const auto offset_y = scale[1];
        const auto offset_z = scale[2];
        const auto voxel_size = scale[3];

        if (h < H && w < W){
            const auto px = pixel[h][w][0];
            const auto py = pixel[h][w][1];
            const auto pz = pixel[h][w][2];
            
            // camera pose tranformation
            auto xe = rotation[0][0] * px + rotation[0][1] * py + rotation[0][2] * pz + translation[0];
            auto ye = rotation[1][0] * px + rotation[1][1] * py + rotation[1][2] * pz + translation[1];
            auto ze = rotation[2][0] * px + rotation[2][1] * py + rotation[2][2] * pz + translation[2];

            xe = (xe - offset_x) / voxel_size;
            ye = (ye - offset_y) / voxel_size;
            ze = (ze - offset_z) / voxel_size;

            // grid shape
            const int vzsize = occ.size(0);
            const int vysize = occ.size(1);
            const int vxsize = occ.size(2);

            // origin
            double xo = translation[0];
            double yo = translation[1];
            double zo = translation[2];

            xo = (xo - offset_x) / voxel_size;
            yo = (yo - offset_y) / voxel_size;
            zo = (zo - offset_z) / voxel_size;

            // locate the voxel where the origin resides
            const int vxo = int(xo);
            const int vyo = int(yo);
            const int vzo = int(zo);

            // origin to end
            const double rx = xe - xo;
            const double ry = ye - yo;
            const double rz = ze - zo;
            double gt_d = sqrt(rx * rx + ry * ry + rz * rz);

            // directional vector
            const double dx = rx / gt_d;
            const double dy = ry / gt_d;
            const double dz = rz / gt_d;

            // In which direction the voxel ids are incremented.
            const int stepX = (dx >= 0) ? 1 : -1;
            const int stepY = (dy >= 0) ? 1 : -1;
            const int stepZ = (dz >= 0) ? 1 : -1;

            // NOTE: new working coords
            int vx = vxo;
            int vy = vyo;
            int vz = vzo;

            // Distance along the ray to the next voxel border from the current position (tMaxX, tMaxY, tMaxZ).
            const double next_voxel_boundary_x = vx + (stepX < 0 ? 0 : 1);
            const double next_voxel_boundary_y = vy + (stepY < 0 ? 0 : 1);
            const double next_voxel_boundary_z = vz + (stepZ < 0 ? 0 : 1);

            // tMaxX, tMaxY, tMaxZ -- distance until next intersection with voxel-border
            // the value of t at which the ray crosses the first vertical voxel boundary
            double tMaxX = (dx!=0) ? (next_voxel_boundary_x - xo)/dx : DBL_MAX; //
            double tMaxY = (dy!=0) ? (next_voxel_boundary_y - yo)/dy : DBL_MAX; //
            double tMaxZ = (dz!=0) ? (next_voxel_boundary_z - zo)/dz : DBL_MAX; //

            // tDeltaX, tDeltaY, tDeltaZ --
            // how far along the ray we must move for the horizontal component to equal the width of a voxel
            // the direction in which we traverse the grid
            // can only be FLT_MAX if we never go in that direction
            const double tDeltaX = (dx!=0) ? stepX/dx : DBL_MAX;
            const double tDeltaY = (dy!=0) ? stepY/dy : DBL_MAX;
            const double tDeltaZ = (dz!=0) ? stepZ/dz : DBL_MAX;

            int3 path[MAX_D];
            double csd[MAX_D];  // cumulative sum of sigma times delta
            double p[MAX_D];  // alpha
            double d[MAX_D];

            // forward raymarching with voxel traversal
            int step = 0;  // total number of voxels traversed
            int count = 0;  // number of voxels traversed inside the voxel grid
            double last_d = 0.0;  // correct initialization

            // voxel traversal raycasting
            bool was_inside = false;
            while (true) {
                bool inside = (0 <= vx && vx < vxsize) &&
                    (0 <= vy && vy < vysize) &&
                    (0 <= vz && vz < vzsize);
                if (inside) {
                    was_inside = true;
                    path[count] = make_int3(vx, vy, vz);
                } else if (was_inside) { // was but no longer inside
                    // we know we are not coming back so terminate
                    break;
                } else { // has not gone inside yet
                    break;
                }
                double _d = 0.0; // The ray distance has traveled before escaping the current voxel cell
                // voxel traversal
                if (tMaxX < tMaxY) {
                    if (tMaxX < tMaxZ) {
                        _d = tMaxX;
                        vx += stepX;
                        tMaxX += tDeltaX;
                    } else {
                        _d = tMaxZ;
                        vz += stepZ;
                        tMaxZ += tDeltaZ;
                    }
                } else {
                    if (tMaxY < tMaxZ) {
                        _d = tMaxY;
                        vy += stepY;
                        tMaxY += tDeltaY;
                    } else {
                        _d = tMaxZ;
                        vz += stepZ;
                        tMaxZ += tDeltaZ;
                    }
                }
                if (inside) {
                    // get sigma at the current voxel
                    const int3 &v = path[count];  // use the recorded index
                    const double _sigma = occ[v.z][v.y][v.x];
                    const double _delta = max(0.0, _d - last_d);  // THIS TURNS OUT IMPORTANT
                    const double sd = _sigma * _delta;
                    if (count == 0) { // the first voxel inside
                        csd[count] = sd;
                        p[count] = 1 - exp(-sd);
                    } else {
                        csd[count] = csd[count-1] + sd;
                        p[count] = exp(-csd[count-1]) - exp(-csd[count]);
                    }
                    // record the traveled distance
                    d[count] = _d;
                    // count the number of voxels we have escaped
                    count ++;
                }
                last_d = _d;
                step ++;
            }

            // the total number of voxels visited should not exceed this number
            assert(count <= MAX_D);

            if (count > 0) {
                // compute the expected ray distance
                double exp_d = 0.0;
                for (int i = 0; i < count; i ++)
                    exp_d += p[i] * d[i];

                // add an imaginary sample at the end point should gt_d exceeds max_d
                double p_out = exp(-csd[count-1]);
                double max_d = d[count-1];

                // p_out is the probability the ray escapes the voxel grid
                exp_d += (p_out * max_d);

                // write the rendered ray distance (max_d)
                dist[h][w] = exp_d * voxel_size;
                depth[h][w] = dist[h][w] * sqrt(rx * rx + ry * ry) / gt_d;
            }
        }
    }


/*
 * input shape
 *  occ         : H x L x W
 *  pixel       : H_img x W_img x 3
 *  rotation    : 3 x 3
 *  translation : 3
 *  scale       : 4
 * output shape
 *   dist     : H_img x W_img
 *   depth    : H_img x W_img
 */

std::vector<torch::Tensor> render_demo_cuda(const torch::Tensor& occ,
                                       const torch::Tensor& pixel,
                                       const torch::Tensor& rotation,
                                       const torch::Tensor& translation,
                                       const torch::Tensor& scale){
    
    const int H = pixel.size(0);
    const int W = pixel.size(1);

    const auto device = occ.device();
    
    const int threads = 32;
    const dim3 threadsPerBlock(threads, threads);
    const dim3 numBlocks((H + threads - 1) / threads, (W + threads - 1) / threads);

    auto dist = -torch::ones({H, W}, device);
    auto depth = torch::zeros({H, W}, device);

    AT_DISPATCH_FLOATING_TYPES(occ.type(), "render_cuda", ([&] {
            render_demo_cuda_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
                occ.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                pixel.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                rotation.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                translation.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                scale.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                dist.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                depth.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
        }));

    cudaDeviceSynchronize();

    return {dist, depth};
}



/* ======================= Differentiable Rendering Occ Grids=======================*/

template <typename scalar_t>
__global__ void render_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> sigma,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> origin,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> pred_dist,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> gt_dist,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_sigma) {

        // batch index
        const auto b = blockIdx.y;

        // ray / point index
        const auto n = blockIdx.x * blockDim.x + threadIdx.x;

        // num of rays / points
        const auto N = points.size(1);

        // we allocated more threads than num_rays
        if (n < N) {
            // grid shape
            const int vzsize = sigma.size(1);
            const int vysize = sigma.size(2);
            const int vxsize = sigma.size(3);

            // origin
            const double xo = origin[b][0];
            const double yo = origin[b][1];
            const double zo = origin[b][2];

            // end point
            const double xe = points[b][n][0];
            const double ye = points[b][n][1];
            const double ze = points[b][n][2];

            // locate the voxel where the origin and end-points resides
            const int vxo = int(xo);
            const int vyo = int(yo);
            const int vzo = int(zo);
            const int vxe = int(xe);
            const int vye = int(ye);
            const int vze = int(ze);

            // current voxel: start from origin
            int vx = vxo;
            int vy = vyo;
            int vz = vzo;

            // origin to end
            const double rx = xe - xo;
            const double ry = ye - yo;
            const double rz = ze - zo;
            double gt_d = sqrt(rx * rx + ry * ry + rz * rz);

            // directional vector
            const double dx = rx / gt_d;
            const double dy = ry / gt_d;
            const double dz = rz / gt_d;

            // In which direction the voxel ids are incremented.
            const int stepX = (dx >= 0) ? 1 : -1;
            const int stepY = (dy >= 0) ? 1 : -1;
            const int stepZ = (dz >= 0) ? 1 : -1;

            // Distance along the ray to the next voxel border from the current position (tMaxX, tMaxY, tMaxZ).
            const double next_voxel_boundary_x = vx + (stepX < 0 ? 0 : 1);
            const double next_voxel_boundary_y = vy + (stepY < 0 ? 0 : 1);
            const double next_voxel_boundary_z = vz + (stepZ < 0 ? 0 : 1);

            // tMaxX, tMaxY, tMaxZ -- distance until next intersection with voxel-border
            // the value of t at which the ray crosses the first vertical voxel boundary
            double tMaxX = (dx!=0) ? (next_voxel_boundary_x - xo)/dx : DBL_MAX; //
            double tMaxY = (dy!=0) ? (next_voxel_boundary_y - yo)/dy : DBL_MAX; //
            double tMaxZ = (dz!=0) ? (next_voxel_boundary_z - zo)/dz : DBL_MAX; //

            // tDeltaX, tDeltaY, tDeltaZ --
            // how far along the ray we must move for the horizontal component to equal the width of a voxel
            // the direction in which we traverse the grid
            // can only be FLT_MAX if we never go in that direction
            const double tDeltaX = (dx!=0) ? stepX/dx : DBL_MAX;
            const double tDeltaY = (dy!=0) ? stepY/dy : DBL_MAX;
            const double tDeltaZ = (dz!=0) ? stepZ/dz : DBL_MAX;

            int3 path[MAX_D];
            double csd[MAX_D];  // cumulative sum of sigma times delta
            double p[MAX_D];  // alpha
            double d[MAX_D];
            double dt[MAX_D];

            // forward raymarching with voxel traversal
            int step = 0;  // total number of voxels traversed
            int count = 0;  // number of voxels traversed inside the voxel grid
            double last_d = 0.0;  // correct initialization

            // voxel traversal raycasting
            bool was_inside = false;
            while (true)
            {
                bool inside = (0 <= vx && vx < vxsize) && (0 <= vy && vy < vysize) && (0 <= vz && vz < vzsize);
                if (inside) { // now inside
                    was_inside = true;
                    path[count] = make_int3(vx, vy, vz);
                } else if (was_inside) { // was inside but no longer
                    // we know we are not coming back so terminate
                    break;
                } else if (last_d > gt_d) { // has not gone inside yet
                    break;
                }

                double _d = 0.0; // Represents the ray distance has traveled before escaping the current voxel cell

                // voxel traversal
                if (tMaxX < tMaxY) {
                    if (tMaxX < tMaxZ) {
                        _d = tMaxX;
                        vx += stepX;
                        tMaxX += tDeltaX;
                    } else {
                        _d = tMaxZ;
                        vz += stepZ;
                        tMaxZ += tDeltaZ;
                    }
                } else {
                    if (tMaxY < tMaxZ) {
                        _d = tMaxY;
                        vy += stepY;
                        tMaxY += tDeltaY;
                    } else {
                        _d = tMaxZ;
                        vz += stepZ;
                        tMaxZ += tDeltaZ;
                    }
                }

                // calculate alpha and T
                if (inside) {
                    // get sigma at the current voxel
                    const int3 &v = path[count];  // use the recorded index
                    const double _sigma = sigma[b][v.z][v.y][v.x];
                    const double _delta = max(0.0, _d - last_d);  // THIS TURNS OUT IMPORTANT
                    const double sd = _sigma * _delta;
                    if (count == 0) { // the first voxel inside
                        csd[count] = sd;
                        p[count] = 1 - exp(-sd);
                    } else {
                        csd[count] = csd[count-1] + sd;
                        p[count] = exp(-csd[count-1]) - exp(-csd[count]);
                    }
                    // record the traveled distance
                    d[count] = _d;
                    dt[count] = _delta;
                    // count the number of voxels we have escaped
                    count ++;
                }

                last_d = _d;
                step ++;
            }
            
            // the total number of voxels visited should not exceed this number
            assert(count <= MAX_D);

            // WHEN THERE IS AN INTERSECTION BETWEEN THE RAY AND THE VOXEL GRID
            if (count > 0) {
                // compute the expected ray distance (weighted sum)
                double exp_d = 0.0;
                for (int i = 0; i < count; i ++)
                    exp_d += p[i] * d[i];

                // add an imaginary sample at the end point should gt_d exceeds max_d
                double p_out = exp(-csd[count-1]);
                double max_d = d[count-1];

                exp_d += (p_out * max_d);
                gt_d = min(gt_d, max_d);

                // write the rendered ray distance (max_d)
                pred_dist[b][n] = exp_d;
                gt_dist[b][n] = gt_d;

                /* backward raymarching */
                double dd_dsigma[MAX_D];
                for (int i = count - 1; i >= 0; i --) {
                    // NOTE: probably need to double check again
                    if (i == count - 1)
                        dd_dsigma[i] = p_out * max_d;
                    else
                        dd_dsigma[i] = dd_dsigma[i+1] - exp(-csd[i]) * (d[i+1] - d[i]);
                }

                for (int i = count - 1; i >= 0; i --)
                    dd_dsigma[i] *= dt[i];

                for (int i = count - 1; i >= 0; i --)
                    dd_dsigma[i] -= dt[i] * p_out * max_d;

                // Use L1 loss
                double dl_dd = (exp_d >= gt_d) ? 1 : -1;
                // apply chain rule
                for (int i = 0; i < count; i ++) {
                    const int3 &v = path[i];
                    grad_sigma[b][v.z][v.y][v.x] += dl_dd * dd_dsigma[i];
                }
            }
        }
    }

/*
 * input shape
 *   sigma      : B x H x L x W
 *   origin   : B x 3
 *   points   : B x N x 4
 * output shape
 *   pred_dist     : B x N
 *   gt_dist     : B x N
 *   grad_sigma : B x H x L x W
 */

std::vector<torch::Tensor> render_cuda(
    torch::Tensor sigma,
    torch::Tensor origin,
    torch::Tensor points) {

    const auto B = points.size(0); // batch size
    const auto N = points.size(1); // num of rays

    const auto device = sigma.device();

    const int threads = 1024;
    const dim3 blocks((N + threads - 1) / threads, B);

    // perform rendering
    auto gt_dist = -torch::ones({B, N}, device);
    auto pred_dist = -torch::ones({B, N}, device);
    auto grad_sigma = torch::zeros_like(sigma);

    AT_DISPATCH_FLOATING_TYPES(sigma.type(), "render_cuda", ([&] {
                render_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    sigma.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                    origin.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    pred_dist.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    gt_dist.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    grad_sigma.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
            }));

    cudaDeviceSynchronize();

    return {pred_dist, gt_dist, grad_sigma};
}



// ======================= pcld mask extractor =====================

template <typename scalar_t>
__global__ void pcldmask_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> bboxes,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> mask) {

        const auto pt_id = blockIdx.x * blockDim.x + threadIdx.x;
        const auto box_id = blockIdx.y;

        const auto N = points.size(0);

        if (pt_id < N) {
            const double x = points[pt_id][0];
            const double y = points[pt_id][1];
            const double z = points[pt_id][2];

            const double ix = bboxes[box_id][1][0] - bboxes[box_id][0][0];
            const double iy = bboxes[box_id][1][1] - bboxes[box_id][0][1];
            const double iz = bboxes[box_id][1][2] - bboxes[box_id][0][2];

            const double jx = bboxes[box_id][3][0] - bboxes[box_id][0][0];
            const double jy = bboxes[box_id][3][1] - bboxes[box_id][0][1];
            const double jz = bboxes[box_id][3][2] - bboxes[box_id][0][2];

            const double kx = bboxes[box_id][4][0] - bboxes[box_id][0][0];
            const double ky = bboxes[box_id][4][1] - bboxes[box_id][0][1];
            const double kz = bboxes[box_id][4][2] - bboxes[box_id][0][2];

            const double vx = x - bboxes[box_id][0][0];
            const double vy = y - bboxes[box_id][0][1];
            const double vz = z - bboxes[box_id][0][2];

            double vi = vx * ix + vy * iy + vz * iz;
            double vj = vx * jx + vy * jy + vz * jz;
            double vk = vx * kx + vy * ky + vz * kz;
            double ii = ix * ix + iy * iy + iz * iz;
            double jj = jx * jx + jy * jy + jz * jz;
            double kk = kx * kx + ky * ky + kz * kz;

            if (vi > 0 && vi < ii && vj > 0 && vj < jj && vk > 0 && vk < kk) {
                mask[pt_id] = 1;
            }
        }
    }


torch::Tensor pcldmask_cuda(const torch::Tensor points, const torch::Tensor bboxes) {
    const auto N = points.size(0); // num of points
    const auto M = bboxes.size(0); // num of bbox

    const auto device = points.device();

    const int threads = 1024;
    const dim3 blocks((N + threads - 1) / threads, M);

    auto mask = torch::zeros({N}, device);

    AT_DISPATCH_FLOATING_TYPES(points.type(), "pcldmask_cuda", ([&] {
                pcldmask_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    points.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    bboxes.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    mask.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
            }));

    cudaDeviceSynchronize();

    return mask;
}