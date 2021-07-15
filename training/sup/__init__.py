def setup(P):

    from .HyMOS_st import train

    dir_name = 'Dataset-'+P.dataset+'_Target-'+P.test_domain+'_Mode-'+P.mode+"_batchK-"+str(P.batch_K)+"_batchP-"+str(P.batch_p)
    if P.iterative:
        dir_name = dir_name + "_iterative"
    if 'st' in P.mode:
        dir_name = dir_name + '_ProbST-'+str(P.adain_probability)

    fname = dir_name

    if P.suffix != "":
        fname += f"_{P.suffix}"

    return train, fname
