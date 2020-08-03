import time
import torch
import json
import os
import numpy as np
import torchvision.utils as vutils
from util.network import calc_gradient_penalty
from util.utils import setNoise, plot_loss, save_model, \
    setNoise_interpolation, latent_lerp_square, setEmb
from config import criterion, material_criterion, color_criterion, \
    perceptual_criterion


def train(opt, dataloader, dataloader_test, device, netD, netG, desc, Noise, optimizerD, optimizerG):
    # for plot
    errD_real_material_total, errD_real_color_total, errD_fake_material_total, errD_fake_color_total \
        = [], [], [], []
    errG_material_total, errG_color_total, epoch_total = [], [], []
    [NZ, noise, fixnoise] = Noise
    for epoch in range(opt.niter):
        # average for each epoch
        errD_real_material_avg, errD_real_color_avg, errD_fake_material_avg, errD_fake_color_avg\
            = 0.0, 0.0, 0.0, 0.0
        errG_material_avg, errG_color_avg, counter = 0.0, 0.0, 0
        real_label = 1
        fake_label = 0
        for i, data in enumerate(dataloader, 0):
            counter += 1
            t0 = time.time()
            # sys.stdout.flush()
            texture, emb, description, _, _, _ = data
            material_label = torch.zeros(opt.batchSize, opt.z_material)
            material_label.copy_(emb[:, :opt.z_material])
            material_label = material_label.long()

            color_label = torch.zeros(opt.batchSize, opt.z_color)
            color_label.copy_(emb[:, opt.z_material:opt.z_material + opt.z_color])
            color_label = color_label.long()

            texture = texture.to(device)
            emb = emb.to(device)
            material_label = material_label.to(device)
            color_label = color_label.to(device)

            # =====================================================================
            # train with real
            netD.zero_grad()
            output_adv, output_material, output_color = netD(texture)
            errD_real_adv = criterion(output_adv, output_adv.detach() * 0 + real_label)
            errD_real_material = material_criterion(output_material.squeeze(), torch.max(material_label, 1)[1])
            errD_real_color = color_criterion(output_color.squeeze(), torch.max(color_label, 1)[1])
            errD_real = errD_real_adv + errD_real_material + errD_real_color

            D_x = output_adv.mean()

            # for average error calculation
            errD_real_material_avg += errD_real_material.item()
            errD_real_color_avg += errD_real_color.item()

            # =====================================================================
            # train with fake
            noise = setNoise(noise)
            fake = netG(noise, emb)

            # output = netD(fake.detach())
            output_adv, output_material, output_color = netD(fake.detach())

            errD_fake_adv = criterion(output_adv, output_adv.detach() * 0 + fake_label)
            errD_fake_material = material_criterion(output_material.squeeze(), torch.max(material_label, 1)[1])
            errD_fake_color = color_criterion(output_color.squeeze(), torch.max(color_label, 1)[1])
            errD_fake = errD_fake_adv + errD_fake_material + errD_fake_color
            D_G_z1 = output_adv.mean()

            # for average error calculation
            errD_fake_material_avg += errD_fake_material.item()
            errD_fake_color_avg += errD_fake_color.item()
            # perceptual loss
            if opt.use_perceptual_loss:
                output_adv_temp, output_material_temp, output_color_temp = netD(texture)
                errD_material_perc = perceptual_criterion(output_material, output_material_temp.detach())
                errD_color_perc = perceptual_criterion(output_color, output_color_temp.detach())

            # total
            errD = errD_real + errD_fake
            if opt.use_perceptual_loss:
                errD += errD_material_perc
                errD += errD_color_perc
            errD.backward()
            if opt.WGAN:
                gradient_penalty = calc_gradient_penalty(netD, texture,
                                                         fake[:texture.shape[0]])  ##for case fewer texture images
                gradient_penalty.backward()
            optimizerD.step()

            if i > 0 and opt.WGAN and i % opt.dIter != 0:
                continue  ## critic steps to 1 GEN steps
            # =====================================================================
            # train G
            netG.zero_grad()
            noise = setNoise(noise)
            fake = netG(noise, emb)

            output_adv, output_material, output_color = netD(fake)
            errG_adv = criterion(output_adv, output_adv.detach() * 0 + real_label)
            errG_material = material_criterion(output_material.squeeze(), torch.max(material_label, 1)[1])
            errG_color = color_criterion(output_color.squeeze(), torch.max(color_label, 1)[1])

            # perceptual loss
            if opt.use_perceptual_loss:
                # output_adv_temp, output_material_temp, output_color_temp = netD(texture, emb)
                output_adv_temp, output_material_temp, output_color_temp = netD(texture)
                errG_material_perc = perceptual_criterion(output_material, output_material_temp.detach())
                errG_color_perc = perceptual_criterion(output_color, output_color_temp.detach())

            # for average error calculation
            errG_material_avg += errG_material.item()
            errG_color_avg += errG_color.item()

            D_G_z2 = output_adv.mean()
            errG = errG_adv + errG_material + errG_color
            # errG = errG_adv

            if opt.use_perceptual_loss:
                errG += errG_material_perc
                errG += errG_color_perc

            errG.backward()
            # optimizerU.step()
            optimizerG.step()

            print('[%d/%d][%d/%d] D(x): %.4f D(G(z)): %.4f / %.4f time %.4f'
                  % (epoch, opt.niter, i, len(dataloader), D_x, D_G_z1, D_G_z2, time.time() - t0))

            ### RUN INFERENCE AND SAVE LARGE OUTPUT MOSAICS
            if epoch % 1000 == 0:
                vutils.save_image(texture, '%s/%s/real_textures_%03d_%s.jpg' % (opt.outputFolder, 'train', epoch, desc),
                                  normalize=True)
                vutils.save_image(fake,
                                  '%s/%s/generated_textures_%03d_%s.jpg' % (opt.outputFolder, 'train', epoch, desc),
                                  normalize=True)
                fixnoise = setNoise(fixnoise)
                netG.eval()
                with torch.no_grad():
                    fakeBig = netG(fixnoise, emb)
                vutils.save_image(fakeBig, '%s/%s/big_texture_%03d_%s.jpg' % (opt.outputFolder, 'train', epoch, desc),
                                  normalize=True)

                netG.train()

                # save description
                description_dict = {}
                for j in range(len(description)):
                    description_dict['{}'.format(j)] = description[j]
                with open('%s/%s/description_%03d_%s.json' % (opt.outputFolder, 'train', epoch, desc), 'w') as f:
                    json.dump(description_dict, f)

                ### evaluation dataset
                for k, data_eval in enumerate(dataloader_test, 0):
                    # handle data
                    texture_eval, emb_eval, description_eval, _, _, _ = data_eval
                    texture_eval = texture_eval.to(device)
                    emb_eval = emb_eval.to(device)
                    netG.eval()
                    noise = setNoise(noise)
                    fake_eval = netG(noise, emb_eval)
                    # vutils.save_image(texture, '%s/real_textures.jpg' % opt.outputFolder,  normalize=True)
                    vutils.save_image(texture_eval,
                                      '%s/%s/real_textures_%03d_%s.jpg' % (opt.outputFolder, 'eval', epoch, desc),
                                      normalize=True)
                    vutils.save_image(fake_eval,
                                      '%s/%s/generated_textures_%03d_%s.jpg' % (opt.outputFolder, 'eval', epoch, desc),
                                      normalize=True)

                    fixnoise = setNoise(fixnoise)

                    with torch.no_grad():
                        # fakeBig_fake=netG(fixnoise)
                        fakeBig_fake = netG(fixnoise, emb_eval)

                    vutils.save_image(fakeBig_fake,
                                      '%s/%s/big_texture_%03d_%s.jpg' % (opt.outputFolder, 'eval', epoch, desc),
                                      normalize=True)

                    netG.train()

                    # save description
                    description_eval_dict = {}
                    for j in range(len(description_eval)):
                        description_eval_dict['{}'.format(j)] = description_eval[j]
                    with open('%s/%s/description_%03d_%s.json' % (opt.outputFolder, 'eval', epoch, desc), 'w') as f:
                        json.dump(description_eval_dict, f)
                    break
        if epoch % 200 == 0:
            # save model
            save_model(netG, epoch, opt.outputFolder, 'netG')
            save_model(netD, epoch, opt.outputFolder, 'netD')
            # save_model(netE, epoch, opt.outputFolder, 'netE')
        epoch_total.append(epoch)

        # average
        errD_real_material_avg = errD_real_material_avg / (counter * opt.batchSize)
        errD_real_color_avg = errD_real_color_avg / (counter * opt.batchSize)
        errD_fake_material_avg = errD_fake_material_avg / (counter * opt.batchSize)
        errD_fake_color_avg = errD_fake_color_avg / (counter * opt.batchSize)
        errG_material_avg = errG_material_avg / (counter * opt.batchSize)
        errG_color_avg = errG_color_avg / (counter * opt.batchSize)

        # append to total
        errD_real_material_total.append(errD_real_material_avg)
        errD_real_color_total.append(errD_real_color_avg)
        errD_fake_material_total.append(errD_fake_material_avg)
        errD_fake_color_total.append(errD_fake_color_avg)
        errG_material_total.append(errG_material_avg)
        errG_color_total.append(errG_color_avg)

        # plot
        plot_loss(0, epoch_total, errD_real_material_total, 'errD_real_material_total',
                  'epoch', 'training error', opt.outputFolder, 'errD_real_material_total.png')
        plot_loss(1, epoch_total, errD_real_color_total, 'errD_real_color_total',
                  'epoch', 'training error', opt.outputFolder, 'errD_real_color_total.png')
        plot_loss(2, epoch_total, errD_fake_material_total, 'errD_fake_material_total',
                  'epoch', 'training error', opt.outputFolder, 'errD_fake_material_total.png')
        plot_loss(3, epoch_total, errD_fake_color_total, 'errD_fake_color_total',
                  'epoch', 'training error', opt.outputFolder, 'errD_fake_color_total.png')
        plot_loss(4, epoch_total, errG_material_total, 'errG_material_total',
                  'epoch', 'training error', opt.outputFolder, 'errG_material_total.png')
        plot_loss(5, epoch_total, errG_color_total, 'errG_color_total',
                  'epoch', 'training error', opt.outputFolder, 'errG_color_total.png')


def test(opt, dataloader, device, netD, netG, Noise, desc):
    # test dataset
    [NZ, noise, fixnoise] = Noise
    for k, data_test in enumerate(dataloader, 0):
        # handle data
        texture_test, emb_test, description_test, _, _, _ = data_test
        texture_test = texture_test.to(device)
        emb_test = emb_test.to(device)
        netG.eval()
        # noise = setNoise(noise, emb_test)
        # fake_test = netG(noise)

        noise = setNoise(noise)
        fake_test = netG(noise, emb_test)

        # vutils.save_image(texture, '%s/real_textures.jpg' % opt.outputFolder,  normalize=True)
        vutils.save_image(texture_test, '%s/%s/real_textures_%03d_%s.jpg' % (opt.outputFolder, 'test', k, desc),
                          normalize=True)
        vutils.save_image(fake_test, '%s/%s/generated_textures_%03d_%s.jpg' % (opt.outputFolder, 'test', k, desc),
                          normalize=True)
        fixnoise = setNoise(fixnoise)
        with torch.no_grad():
            # fakeBig_fake=netG(fixnoise)
            fakeBig_fake = netG(fixnoise, emb_test)

        vutils.save_image(fakeBig_fake, '%s/%s/big_texture_%03d_%s.jpg' % (opt.outputFolder, 'test', k, desc),
                          normalize=True)
        # save description
        description_test_dict = {}
        for j in range(len(description_test)):
            description_test_dict['{}'.format(j)] = description_test[j]
        with open('%s/%s/description_%03d_%s.json' % (opt.outputFolder, 'test', k, desc), 'w') as f:
            json.dump(description_test_dict, f)


def save_embedding(opt, dataloader_test, device, netD):
    material_embeddings = np.zeros((opt.batchSize * len(dataloader_test), 19 * 5 * 5 + 1))
    color_embeddings = np.zeros((opt.batchSize * len(dataloader_test), 12 * 5 * 5 + 1))
    # save embedding
    for i, data in enumerate(dataloader_test, 0):
        # handle data
        texture, emb, description, material_label, color_label, apply_label = data
        texture = texture.to(device)
        emb = emb.to(device)
        netD.eval()

        _, output_material, output_color = netD(texture)
        # copy to the numpy array
        material_embeddings[i * opt.batchSize:(i + 1) * opt.batchSize, :-1] = \
            output_material.view(opt.batchSize, -1).detach().cpu().numpy()
        material_embeddings[i * opt.batchSize:(i + 1) * opt.batchSize, -1] = \
            material_label.view(opt.batchSize).cpu().numpy()

        color_embeddings[i * opt.batchSize:(i + 1) * opt.batchSize, :-1] = \
            output_color.view(opt.batchSize, -1).detach().cpu().numpy()
        color_embeddings[i * opt.batchSize:(i + 1) * opt.batchSize, -1] = \
            color_label.view(opt.batchSize).cpu().numpy()
    # save the numpy array
    np.save('../material_embeddings.npy', material_embeddings)
    np.save('../color_embeddings.npy', color_embeddings)


def interpolation(opt, device, netG, Noise):
    # emb 1
    mp1 = 1
    cp1 = 2
    ap1 = 0
    # emb 2
    mp2 = 10
    cp2 = 5
    ap2 = 0
    counter = 4
    [NZ, noise, fixnoise] = Noise
    # names of image and text
    # img_name = 'textures_interpolation_square'
    # text_name = 'description_square'
    img_name = '%05d' % (counter)
    text_name = '%05d' % (counter)
    # generate two input noise+emb
    noise1, noise2, emb1, emb2, description = setNoise_interpolation(noise, \
                                                                     mp1=mp1, mp2=mp2, cp1=cp1, cp2=cp2, ap1=ap1,
                                                                     ap2=ap2)
    noise1 = noise1.to(device)
    noise2 = noise2.to(device)
    emb1 = emb1.to(device)
    emb2 = emb2.to(device)
    netG.eval()

    # imgs = latent_lerp(netG, noise1, noise2, nb_frames=16)
    imgs = latent_lerp_square(netG, noise1, noise2, emb1, emb2, nb_frames=64)
    # save image
    try:
        os.makedirs(os.path.join(opt.outputFolder, 'interpolation'))
    except OSError:
        pass
    vutils.save_image(imgs, '%s/%s/%s.jpg' % (opt.outputFolder, 'interpolation', img_name), normalize=True)
    # save text
    with open('%s/%s/%s.txt' % (opt.outputFolder, 'interpolation', text_name), 'w') as f:
        f.write(description)
    counter += 1


def all_composition(opt, device, netG, Noise):
    counter = 500
    [NZ, noise, fixnoise] = Noise
    for ap1 in range(2):
        for mp1 in range(19):
            for cp1 in range(12):
                # names of image and text
                # img_name = 'textures_interpolation_square'
                # text_name = 'description_square'
                img_name = 'generated_textures_%03d.jpg' % (counter)
                text_name = 'description_%03d.json' % (counter)

                # generate emb
                emb, description = setEmb(mp1, cp1, ap1)

                noise = setNoise(noise)
                noise = noise.to(device)
                noise = noise[torch.arange(noise.size(0))==0]
                emb = emb.unsqueeze(0)
                emb = emb.to(device)
                netG.eval()

                # imgs = latent_lerp(netG, noise1, noise2, nb_frames=16)
                fake = netG(noise, emb)

                # save image
                try:
                    os.makedirs(os.path.join(opt.outputFolder, 'all_composition'))
                except OSError:
                    pass

                vutils.save_image(fake, '%s/%s/%s' % (opt.outputFolder, 'all_composition', img_name), normalize=True)

                # save text
                with open('%s/%s/%s' % (opt.outputFolder, 'all_composition', text_name), 'w') as f:
                    json.dump(description, f)
                counter += 1
