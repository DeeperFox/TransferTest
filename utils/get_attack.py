import torchattacks


def get_attack(cfg, model, device,random_image, random_label):
    # Attack
    if cfg.method == 'fgsm':
        attack_method = torchattacks.FGSM(model=model, eps=cfg.ADV.eps)
    elif cfg.method == 'pgd':
        attack_method = torchattacks.PGD(model=model, eps=cfg.ADV.eps, alpha=cfg.ADV.alpha, steps=cfg.ADV.iters)
    elif cfg.method == 'mi-fgsm':
        attack_method = torchattacks.MIFGSM(model=model, eps=cfg.ADV.eps, alpha=cfg.ADV.alpha, steps=cfg.ADV.iters,
                                            decay=cfg.ADV.MI.decay)
    elif cfg.method == 'di-fgsm':
        attack_method = torchattacks.DIFGSM(model=model, eps=cfg.ADV.eps, alpha=cfg.ADV.alpha, steps=cfg.ADV.iters,
                                            decay=cfg.ADV.DI.decay, resize_rate=cfg.ADV.DI.resize_rate,
                                            diversity_prob=cfg.ADV.DI.diversity_prob)
    elif cfg.method == 'pi-fgsm':
        from attack import PI_FGSM
        attack_method = PI_FGSM.PI_FGSM(model=model, eps=cfg.ADV.eps, alpha=cfg.ADV.alpha, iters=cfg.ADV.iters,
                                        amplification=cfg.ADV.PI.amplification, kern_size=cfg.ADV.PI.kern_size)
    elif cfg.method == 'ti-fgsm':
        attack_method = torchattacks.TIFGSM(model=model, eps=cfg.ADV.eps, alpha=cfg.ADV.alpha, steps=cfg.ADV.iters,
                                            decay=cfg.ADV.TI.decay, len_kernel=cfg.ADV.TI.len_kernel,
                                            nsig=cfg.ADV.TI.nsig, resize_rate=cfg.ADV.TI.resize_rate,
                                            diversity_prob=cfg.ADV.TI.diversity_prob)
    elif cfg.method == 'ni-fgsm':
        attack_method = torchattacks.NIFGSM(model=model, eps=cfg.ADV.eps, alpha=cfg.ADV.alpha, steps=cfg.ADV.iters,
                                            decay=cfg.ADV.NI.decay)
    elif cfg.method == 'sini-fgsm':
        attack_method = torchattacks.SINIFGSM(model=model, eps=cfg.ADV.eps, alpha=cfg.ADV.alpha, steps=cfg.ADV.iters,
                                              decay=cfg.ADV.SINI.decay, m=cfg.ADV.SINI.m)
    elif cfg.method == 'vmi-fgsm':
        attack_method = torchattacks.VMIFGSM(model=model, eps=cfg.ADV.eps, alpha=cfg.ADV.alpha, steps=cfg.ADV.iters,
                                             N=cfg.ADV.VMI.N, beta=cfg.ADV.VMI.beta)
    elif cfg.method == 'vni-fgsm':
        attack_method = torchattacks.VNIFGSM(model=model, eps=cfg.ADV.eps, alpha=cfg.ADV.alpha, steps=cfg.ADV.iters,
                                             N=cfg.ADV.VNI.N, beta=cfg.ADV.VNI.beta)
    elif cfg.method == 'admix':
        from attack import AdMix
        attack_method = AdMix.Admix(model=model, eps=cfg.ADV.eps, alpha=cfg.ADV.alpha, iters=cfg.ADV.iters,
                                    m_1=cfg.ADV.Admix.m_1, m_2=cfg.ADV.Admix.m_2, eta=cfg.ADV.Admix.eta,
                                    decay=cfg.ADV.Admix.decay)
    elif cfg.method == 'smi-fgsm':
        from attack import SMI_FGSM
        attack_method = SMI_FGSM.SMI_FGSM(model=model, eps=cfg.ADV.eps, alpha=cfg.ADV.alpha, iters=cfg.ADV.iters,
                                          decay=cfg.ADV.DI.decay, resize_rate=cfg.ADV.DI.resize_rate,
                                          diversity_prob=cfg.ADV.DI.diversity_prob)
    elif cfg.method == 'di-fgsm2':
        from attack import DI_FGSM2
        attack_method = DI_FGSM2.DI_FGSM(model=model, eps=cfg.ADV.eps, alpha=cfg.ADV.alpha, steps=cfg.ADV.iters,
                                         decay=cfg.ADV.DI.decay, resize_rate=cfg.ADV.DI.resize_rate,
                                         diversity_prob=cfg.ADV.DI.diversity_prob)

    elif cfg.method == 'ours':
        from attack import OURS
        attack_method = OURS.CI_FGSM(model=model, eps=cfg.ADV.eps, alpha=cfg.ADV.alpha, steps=cfg.ADV.iters,
                                         decay=cfg.ADV.DI.decay, resize_rate=cfg.ADV.DI.resize_rate,
                                         diversity_prob=cfg.ADV.DI.diversity_prob,random_image=random_image,
                                   random_label=random_label)
    #################################################################
    #                            Test                               #
    #################################################################
    else:
        raise NotImplemented

    return attack_method
