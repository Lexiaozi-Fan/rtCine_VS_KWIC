%% add path
clc
clear all

addpath /Functions/

%% Load your data and organize as below: 1: nx, 2:ny ,3: time dimension ,4: slice dimension, 5: coil dimension
[sr,sth,sz,slc,lset] = size(kSpace);


N= 5;tau = (1.0+sqrt(5.0))/2.0; % tiny golden angle, change accordingly
golden_angle = pi / (tau+N-1);


clear recon_cs_TV;

for slice_counter= 1:slc
    display(num2str(slice_counter));
    Coil1=squeeze(double(kSpace(:,:,:,slice_counter,:)));
    if lset>8
        Coil1 = coil_compression(Coil1,8); % coil compression to 8 coils
    end
    [mx,my,mz,mc,ms] = size(Coil1);

    % remove dummy scan and trajectory correction scan as we use RING
    % method for trajectory correction
    Coil1 = Coil1(:,:,1:mz-4,:);
    clear Coil_temp
    [reduced_non_k_space,kdata2,mask2]=pre_interp_clean_mod(Coil1,slice_counter,N,timeframe);
    kdata2 = kdata2(:,:,35:sz-4,:);
    mask2 = mask2(:,:,35:sz-4,:);
   
    
    %Coil maps generated here
    DC=squeeze(sum(kdata2,3));
    SM=squeeze(sum(mask2,3));
    SM(find(SM==0))=1;
    for ii=1:mc
        DC(:,:,ii)=DC(:,:,ii)./SM;
    end
    
    %This ref is the unpaired FFT for the coil sensitivity maps. The
    %unpaired FFT for the kdata is withinEmat_2DPERF.mtimes
    ref=ifft2c_mri(DC);
    [dummy,b1]=adapt_array_2d_st2(ref);
    b1=b1/max(b1(:));
    
    [nr,np,nt,nc] = size(Coil1);
    count = 1;
    for jj = 1:nt
        for kk = 1:np
            kdata_corrected(:,count,:) = squeeze(Coil1(:,kk,jj,:));
            count = count +1;
        end
    end
    ntviews = nt*np;
    nspokes= 6; % number of radial spokes used for reconstructing individual image
    nx = size(kdata_corrected,1);
    
    Angles = golden_angle*(0:nt*np-1);
    Angles = mod(Angles,pi);
    ray1 = -0.5:1/nx:0.5-1/nx;    
    k = zeros(sr,length(Angles));
    for ii = 1:length(Angles)
        k(:,ii) = ray1.*exp(1i*Angles(ii));
    end
    w = abs(real(k) + 1i*imag(k))./max(abs(real(k(:))  + 1i*imag(k(:)))); w(isnan(w))=1;
    w( find( w == 0 ) ) = 1.1102e-16;
    % data dimensions
    clear kdata;
    %% RING trajectory correction
    nt=floor(ntviews/nspokes);
    kdata_corr = ring_correction_mod(kdata_corrected,270:ntviews,N); % ring correction kdata_corr angle info
    
    k=k(:,1:nt*nspokes);
    w=w(:,1:nt*nspokes);
    angle_3 = reshape(Angles(1:nspokes*nt),[nspokes nt]);
    % sort the data into a time-series
    clear kdatau,clear ku,clear kdatau_venc
    for ii= 3:nt-2
        kdatau(:,:,:,ii)=kdata_corrected(:,(ii-1)*nspokes+1-9:ii*nspokes+9,:); % view sharing 18 radial spokes
        ku(:,:,ii)=kdata_corr(:,(ii-1)*nspokes+1-9:ii*nspokes+9); 
    end
    [nx ny nc nt] = size(kdatau);
    wu = ones(nx, ny, nt);
    % applied KWIC filtering to zero the center of shared radial spokes
    wu(nx/2-4:nx/2+6,1:9,:) = 0; 
    wu(nx/2-4:nx/2+6,16:24,:) = 0;
    wu( find( wu == 0 ) ) = eps;
    ku = 1i.*real(ku)+imag(ku);
    ku_dummy = ku(:,:,35:size(ku,3));
    kdatau_dummy = kdatau(:,:,:,35:size(ku,3));
    wu_dummy = wu(:,:,35:size(ku,3));
    param.E = MCNUFFT_GPU(ku_dummy,wu_dummy,b1);
    param.y=kdatau_dummy;
    recon_nufft=param.E'*param.y;
%% CS recon    
    param.TV = TV_Temp();
    param.W1 = TempPCA();
    param.nite = 9;
    param.display = 1;
    param.TVWeight = max(abs(ref(:))).* 0.005; % TTV weight
    param.L1Weight1 = param.TVWeight*0.25;
    recon_cs = recon_nufft;
    fprintf('\n GRASP reconstruction \n')
    tic;
    for n = 1 : 3
        recon_cs = CSL1NlCg_PCA(recon_cs,param);
    end
    toc
    recon_cs_TV(:,:,:,slice_counter) = fliplr(rot90(single(recon_cs)));
end


save([path  'Recon.mat'],'recon_cs_TV','-v7.3')


