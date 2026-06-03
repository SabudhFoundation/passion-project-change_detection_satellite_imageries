import tensorflow as tf
from tensorflow.keras import layers, Model

K_WEIGHT_MAX = 10.0

def conv3x3(x, filters, name):
    x = layers.Conv2D(filters, kernel_size=3, padding='same', name=name)(x)
    x = layers.Activation('relu')(x)
    return x


def conv1x1(x, filters, name):
    x = layers.Conv2D(filters, kernel_size=1, padding='same', name=name)(x)
    x = layers.Activation('relu')(x)
    return x


def conv2x2(x, filters, name):
    x = layers.Conv2D(filters, kernel_size=2, padding='same', name=name)(x)
    x = layers.Activation('relu')(x)
    return x


def encoder_stage1(t1, t2):
    shared_conv1 = layers.Conv2D(16, 3, padding='same', activation='relu', name="s1_conv1")
    shared_conv2 = layers.Conv2D(16, 3, padding='same', activation='relu', name="s1_conv2")

    f1 = shared_conv1(t1)
    f2 = shared_conv1(t2)

    diff = layers.Subtract(name="s1_diff")([f1, f2])

    cr = conv1x1(diff, 16, name="s1_cr")

    f1 = shared_conv2(f1)
    f2 = shared_conv2(f2)

    skip1 = layers.Concatenate(name="s1_skip1")([f1, cr])
    skip2 = layers.Concatenate(name="s1_skip2")([f2, cr])


    pool = layers.MaxPool2D(2, name="s1_pool")
    out1 = pool(skip1)
    out2 = pool(skip2)

    return skip1, skip2, out1, out2

def encoder_stage2(x1, x2):

    shared_conv1 = layers.Conv2D(32, 3, padding='same', activation='relu', name="s2_conv1")
    shared_conv2 = layers.Conv2D(32, 3, padding='same', activation='relu', name="s2_conv2")

    f1 = shared_conv1(x1)
    f2 = shared_conv1(x2)

    diff = layers.Subtract(name="s2_diff")([f1, f2])

    cr   = conv1x1(diff, 32, name="s2_cr")

    f1 = shared_conv2(f1)
    f2 = shared_conv2(f2)

    skip1 = layers.Concatenate(name="s2_skip1")([f1, cr])
    skip2 = layers.Concatenate(name="s2_skip2")([f2, cr])

    pool = layers.MaxPool2D(2, name="s2_pool")
    out1 = pool(skip1)
    out2 = pool(skip2)

    return skip1, skip2, out1, out2

def encoder_stage3(x1, x2):

    shared_conv1 = layers.Conv2D(64, 3, padding='same', activation='relu', name="s3_conv1")
    shared_conv2 = layers.Conv2D(64, 3, padding='same', activation='relu', name="s3_conv2")
    shared_conv3 = layers.Conv2D(64, 3, padding='same', activation='relu', name="s3_conv3")

    f1 = shared_conv1(x1)
    f2 = shared_conv1(x2)

    diff = layers.Subtract(name="s3_diff")([f1, f2])

    cr   = conv1x1(diff, 64, name="s3_cr")

    f1 = shared_conv2(f1)
    f1 = shared_conv3(f1)
    f2 = shared_conv2(f2)
    f2 = shared_conv3(f2)



    skip1 = layers.Concatenate(name="s3_skip1")([f1, cr])
    skip2 = layers.Concatenate(name="s3_skip2")([f2, cr])

    pool = layers.MaxPool2D(2, name="s3_pool")
    out1 = pool(skip1)
    out2 = pool(skip2)

    return skip1, skip2, out1, out2


def encoder_stage4(x1, x2):

    shared_conv1 = layers.Conv2D(128, 3, padding='same', activation='relu', name="s4_conv1")
    shared_conv2 = layers.Conv2D(128, 3, padding='same', activation='relu', name="s4_conv2")
    shared_conv3 = layers.Conv2D(128, 3, padding='same', activation='relu', name="s4_conv3")

    f1 = shared_conv1(x1)
    f2 = shared_conv1(x2)

    diff = layers.Subtract(name="s4_diff")([f1, f2])

    cr   = conv1x1(diff, 128, name="s4_cr")

    f1 = shared_conv2(f1)
    f1 = shared_conv3(f1)


    f2 = shared_conv2(f2)
    f2 = shared_conv3(f2)

    feat1 = layers.Concatenate(name="s4_feat1")([f1, cr])
    feat2 = layers.Concatenate(name="s4_feat2")([f2, cr])

    r1 = conv1x1(feat1, 64, name="s4_reduce1")
    r2 = conv1x1(feat2, 64, name="s4_reduce2")

    diff_r = layers.Subtract(name="s4_diff_final")([r1, r2])
    f_enc = layers.Concatenate(name="s4_enc_out")([r1, r2, diff_r])

    return f_enc




def nspp_block(f_enc):
    """
    NSPP Block — paper-faithful implementation of Equations 4–7.

    Input : f_enc    → (B, H, W, C)   C=192 for 128×128 input (H=W=16)
    Output: fnew_SPP → (B, H, W, C)   same spatial size and channel count

    Eq. 4 — Pooling block per path (stride s ∈ {2,4,8,16}):
        fp = ReLU( strided_conv3x3(f_enc, s) + avgpool_s(f_enc) * w^{1×1} )
        Shape: (B, H/s, W/s, C/4)

    Eq. 5 — Global context: fg_o = reduce_mean(fp) * w^{1×1}_g
        Shape: (B, 1, 1, C/4)

    Eq. 6 — Upsample back to (H, W):
        fp_o = ReLU( fg_o ⊛ w^{3×3}_t )  with stride=(H, W)
        Implemented as UpSampling2D + Conv2D to support dynamic shapes
        (fix #2 — Conv2DTranspose(stride=(H,W)) breaks when H is None).

    Eq. 7 — Aggregate:
        fnew_SPP = [fp_o1, fp_o2, fp_o3, fp_o4, f_enc] * w^{1×1}
    """
    C  = f_enc.shape[-1]   
    c4 = C // 4            

    
    H_static = f_enc.shape[1]   
    W_static = f_enc.shape[2]

    path_outputs = []

    for stride in [2, 4, 8, 16]:

        
        b_conv = layers.SeparableConv2D(
        c4, 3, strides=stride, padding='same',
        name=f"nspp_strided_s{stride}"
        )(f_enc)   

        
        b_avg = layers.AveragePooling2D(
            pool_size=stride, strides=stride, padding='same',
            name=f"nspp_avg_s{stride}"
        )(f_enc)
        b_avg = layers.Conv2D(
            c4, 1, padding='same',
            name=f"nspp_avg_pw_s{stride}"
        )(b_avg)   # (B, H/s, W/s, C/4)

        fp = layers.Add(name=f"nspp_add_s{stride}")([b_conv, b_avg])
        fp = layers.Activation('relu', name=f"nspp_relu_s{stride}")(fp)
        # fp: (B, H/s, W/s, C/4)

        
        fg = layers.GlobalAveragePooling2D(
            name=f"nspp_gap_s{stride}"
        )(fp)   # (B, C/4)
        fg = layers.Reshape(
            (1, 1, c4), name=f"nspp_reshape_s{stride}"
        )(fg)   
        fg = layers.Conv2D(
            c4, 1, padding='same',
            name=f"nspp_gpw_s{stride}"
        )(fg)   

        
        if H_static is not None and W_static is not None:
            fp_out = layers.UpSampling2D(
            size=(H_static, W_static), interpolation='bilinear',
            name=f"nspp_up_s{stride}"
            )(fg)
            fp_out = layers.Conv2D(
            c4, 3, padding='same', activation='relu',
            name=f"nspp_post_up_conv_s{stride}"
            )(fp_out)
        else:
            
            fp_out = layers.Lambda(
                lambda inputs: tf.image.resize(
                    inputs[0],
                    (tf.shape(inputs[1])[1], tf.shape(inputs[1])[2]),
                    method='bilinear'
                ),
                name=f"nspp_up_dyn_s{stride}"
            )([fg, f_enc])
            fp_out = layers.Conv2D(
                c4, 3, padding='same', activation='relu',
                name=f"nspp_post_up_conv_s{stride}"
            )(fp_out)
        

        path_outputs.append(fp_out)


    cat = layers.Concatenate(name="nspp_cat")(path_outputs + [f_enc])
    out = layers.Conv2D(
        C, 1, padding='same', activation='relu',
        name="nspp_out"
    )(cat)   

    return out


def decoder_stage1(x, skip1_s3, skip2_s3):

    x = layers.UpSampling2D(2, name="d1_up")(x)
    x = conv2x2(x, 64, name="d1_up_conv")

    x = layers.Concatenate(name="d1_cat")([x, skip1_s3, skip2_s3])

    x = conv3x3(x, 64, name="d1_conv1")
    x = conv3x3(x, 64, name="d1_conv2")
    x = conv3x3(x, 32, name="d1_conv3")
    x = layers.BatchNormalization(name="d1_bn")(x)
    return x


def decoder_stage2(x, skip1_s2, skip2_s2):

    x = layers.UpSampling2D(2, name="d2_up")(x)
    x = conv2x2(x, 32, name="d2_up_conv")


    x = layers.Concatenate(name="d2_cat")([x, skip1_s2, skip2_s2])

    x = conv3x3(x, 32, name="d2_conv1")
    x = conv3x3(x, 16, name="d2_conv2")
    x = layers.BatchNormalization(name="d2_bn")(x)
    return x


def decoder_stage3(x, skip1_s1, skip2_s1):

    x = layers.UpSampling2D(2, name="d3_up")(x)
    x = conv2x2(x, 16, name="d3_up_conv")

    x = layers.Concatenate(name="d3_cat")([x, skip1_s1, skip2_s1])

    x = conv3x3(x, 16, name="d3_conv1")
    x = layers.BatchNormalization(name="d3_bn")(x)
    return x


def build_ucdnet(input_shape=(512, 512, 13), num_classes=2):


    T1 = layers.Input(shape=input_shape, name="T1")
    T2 = layers.Input(shape=input_shape, name="T2")


    skip1_s1, skip2_s1, o1, o2 = encoder_stage1(T1, T2)
    skip1_s2, skip2_s2, o1, o2 = encoder_stage2(o1, o2)
    skip1_s3, skip2_s3, o1, o2 = encoder_stage3(o1, o2)
    f_enc = encoder_stage4(o1, o2)


    f_nspp = nspp_block(f_enc)


    d1 = decoder_stage1(f_nspp, skip1_s3, skip2_s3)
    d2 = decoder_stage2(d1,     skip1_s2, skip2_s2)
    d3 = decoder_stage3(d2,     skip1_s1, skip2_s1)

    output = layers.Conv2D(num_classes, 1, padding='same',
                            activation='softmax', name="change_map")(d3)


    model = Model(inputs=[T1, T2], outputs=output, name="UCDNet")
    return model




# LOSS FUNCTION


def ucdnet_loss(class_weights=(0.1, 0.9), alpha=0.3, beta=0.7):

    w = tf.constant(list(class_weights), dtype=tf.float32)
 
    def loss(y_true, y_pred):
        y_pred_c = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
 
        # ── Eq. 14: WCCE ──────────────────────────────────────────────────
        # L_wcce = -1/K * sum_{i,j,k} w_k * Y_ijk * log(p_ijk)
        # Use reduce_mean over pixels, reduce_sum over classes (K=2)
        K    = tf.cast(tf.shape(y_true)[-1], tf.float32)
        wcce = -tf.reduce_mean(
            tf.reduce_sum(y_true * tf.math.log(y_pred_c) * w, axis=-1)
        )
        # Divide by K as per paper Eq. 14
        wcce = wcce / K
 
        # ── Eq. 16: Soft kappa ────────────────────────────────────────────
        # changed class = channel index 1 (positive class)
        p = tf.reshape(y_pred_c[..., 1], [-1])   # predicted prob of change
        t = tf.reshape(y_true[..., 1],   [-1])   # ground truth change mask
 
        # Unweighted counts — used for pe (expected agreement)
        TP_raw = tf.reduce_sum(t * p)
        TN_raw = tf.reduce_sum((1.0 - t) * (1.0 - p))
        FP_raw = tf.reduce_sum((1.0 - t) * p)
        FN_raw = tf.reduce_sum(t * (1.0 - p))
        N_raw  = TP_raw + TN_raw + FP_raw + FN_raw + 1e-7
 
        # Weighted counts — alpha on TP, beta on TN, used for po (observed agreement)
        TP_w = tf.reduce_sum(alpha * t * p)
        TN_w = tf.reduce_sum(beta  * (1.0 - t) * (1.0 - p))
 
        # Observed agreement (weighted)
        po = (TP_w + TN_w) / N_raw
 
        # Expected agreement (unweighted raw counts)
        pe = (
            (TP_raw + FP_raw) * (TP_raw + FN_raw) +
            (TN_raw + FN_raw) * (TN_raw + FP_raw)
        ) / (N_raw * N_raw + 1e-7)
 
        # Cohen's kappa: ka = (po - pe) / (1 - pe)
        ka = (po - pe) / (1.0 - pe + 1e-7)
 
        #  Kappa loss 
        L_kappa = 1.0 - ka
 
        # Log-Cosh smoothed kappa loss 
        L_modified_kappa = tf.math.log(tf.math.cosh(L_kappa + 1e-7))
 
        # Eq. 17: Adaptive kappa weight — clipped for stability 
        # k_weight = 1 + L_kappa / ka
        k_weight = 1.0 + L_kappa / (ka + 1e-7)
        k_weight = tf.clip_by_value(k_weight, 0.0, K_WEIGHT_MAX)
 
        #  Combined loss 
        return wcce + k_weight * L_modified_kappa
 
    loss.__name__ = "ucdnet_loss"
    return loss
 
 

# COMPILE

 
def compile_ucdnet(model, class_weights=(0.1, 0.9), learning_rate=1e-4,
                   kappa_alpha=0.5, kappa_beta=0.5):
    """
    Compile UCDNet with Adam + paper loss.
    Paper: Adam lr=0.0001, 30 epochs, batch_size=1.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=ucdnet_loss(class_weights=class_weights,
                         alpha=kappa_alpha,
                         beta=kappa_beta),
        metrics=['accuracy']
    )
    return model


# QUICK TEST


if __name__ == "__main__":

    print("Building UCDNet ...")
    model = build_ucdnet(input_shape=(512, 512, 13), num_classes=2)
    compile_ucdnet(model)
    model.summary()

    print("\nRunning a quick forward pass with dummy data ...")
    import numpy as np
    T1_dummy = np.random.rand(1, 512, 512, 13).astype("float32")
    T2_dummy = np.random.rand(1, 512, 512, 13).astype("float32")

    output = model.predict([T1_dummy, T2_dummy], verbose=0)

    print(f"Input  shape : {T1_dummy.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Total params : {model.count_params():,}")
    assert output.shape == (1, 512, 512, 2), "Output shape mismatch!"
    print("\nDone! Model built successfully.")