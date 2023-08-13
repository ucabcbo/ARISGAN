#!/bin/sh

# ./sched.sh 8 50 s5lw_inmask/alt12/dmnet
# ./sched_l.sh 8 50 s5lw_inmask/alt12/dsen2w
# ./sched.sh 8 50 s5lw_inmask/alt12/pix2pix
# ./sched.sh 8 50 s5lw_inmask/alt12/psgan
# ./sched.sh 8 50 s5lw_inmask/alt12/sis2
# ./sched.sh 8 50 s5lw_inmask/alt12/sis2-1
# ./sched.sh 8 50 s5lw_inmask/alt12/sis2-2
# ./sched_l.sh 8 50 s5lw_inmask/alt12/sis2-3
# ./sched.sh 8 50 s5lw_inmask/alt12/sis3
# ./sched.sh 8 50 s5lw_inmask/alt12/sis3-1
# ./sched.sh 8 50 s5lw_inmask/alt12/sis3-2
# ./sched.sh 8 50 s5lw_inmask/alt12/sis3-3
# ./sched.sh 8 50 s5lw_inmask/alt12/sis3-a
# ./sched.sh 8 50 s5lw_inmask/alt12/sis3-b
# ./sched.sh 8 50 s5lw_inmask/alt12/sis3-c
# ./sched.sh 8 50 s5lw_inmask/alt12/sis3-d
# ./sched.sh 8 50 s5lw_inmask/alt12/sis4
# ./sched.sh 8 50 s5lw_inmask/alt12/sis5
# ./sched.sh 8 50 s5lw_inmask/alt12/sis6
# ./sched.sh 8 50 s5lw_inmask/alt12/sis10
# ./sched_l.sh 8 50 s5lw_inmask/alt12/sis11
# ./sched.sh 8 50 s5lw_inmask/alt12/sis12
# ./sched.sh 8 50 s5lw_inmask/alt12/srgan
# ./sched.sh 8 50 s5lw_inmask/alt12/srs3
# ./sched_l.sh 8 50 s5lw_inmask/alt12/tarsgan_b8
# ./sched_l.sh 8 50 s5lw_inmask/alt12/sis22_b4
# ./sched_l.sh 8 50 s5lw_inmask/alt12/sis23_b1
# ./sched_l.sh 8 50 s5lw_inmask/alt12/sis25_b8
# ./sched_l.sh 8 50 s5lw_inmask/alt12/sis26_b8

# ./sched.sh 8 50 s5lw_inmask/cur/dmnet
# ./sched.sh 8 50 s5lw_inmask/cur/dsen2w
# ./sched.sh 8 50 s5lw_inmask/cur/pix2pix
# ./sched.sh 8 50 s5lw_inmask/cur/psgan
# ./sched.sh 8 50 s5lw_inmask/cur/sis2
# ./sched.sh 8 50 s5lw_inmask/cur/sis2-1
# ./sched.sh 8 50 s5lw_inmask/cur/sis2-2
# ./sched.sh 8 50 s5lw_inmask/cur/sis2-3
# ./sched.sh 8 50 s5lw_inmask/cur/sis3
# ./sched.sh 8 50 s5lw_inmask/cur/sis3-1
# ./sched.sh 8 50 s5lw_inmask/cur/sis3-2
# ./sched.sh 8 50 s5lw_inmask/cur/sis3-3
# ./sched.sh 8 50 s5lw_inmask/cur/sis3-a
# ./sched.sh 8 50 s5lw_inmask/cur/sis3-b
# ./sched.sh 8 50 s5lw_inmask/cur/sis3-c
# ./sched.sh 8 50 s5lw_inmask/cur/sis3-d
# ./sched.sh 8 50 s5lw_inmask/cur/sis4
# ./sched.sh 8 50 s5lw_inmask/cur/sis5
# ./sched.sh 8 50 s5lw_inmask/cur/sis6
# ./sched.sh 8 50 s5lw_inmask/cur/sis10
# ./sched_l.sh 8 50 s5lw_inmask/cur/sis11
# ./sched.sh 8 50 s5lw_inmask/cur/sis12
# ./sched.sh 8 50 s5lw_inmask/cur/srgan
# ./sched.sh 8 50 s5lw_inmask/cur/srs3
# ./sched_l.sh 8 50 s5lw_inmask/cur/tarsgan_b8
# ./sched_l.sh 8 50 s5lw_inmask/cur/sis22_b4
# ./sched_l.sh 8 50 s5lw_inmask/cur/sis23_b1
# ./sched_l.sh 8 50 s5lw_inmask/cur/sis25_b8
# ./sched_l.sh 8 50 s5lw_inmask/cur/sis26_b8

# ./sched.sh 8 50 s5lw_notinmask/alt12/dmnet
# ./sched_l.sh 8 50 s5lw_notinmask/alt12/dsen2w
# ./sched.sh 8 50 s5lw_notinmask/alt12/pix2pix
# ./sched.sh 8 50 s5lw_notinmask/alt12/psgan
# ./sched_l.sh 8 50 s5lw_notinmask/alt12/sis2
# ./sched_l.sh 8 50 s5lw_notinmask/alt12/sis2-1
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis2-2
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis2-3
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis3
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis3-1
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis3-2
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis3-3
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis3-a
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis3-b
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis3-c
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis3-d
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis4
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis5
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis6
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis10
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis11
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis12
# ./sched.sh 8 50 s5lw_notinmask/alt12/srgan
# ./sched.sh 8 50 s5lw_notinmask/alt12/srs3
# ./sched_l.sh 8 50 s5lw_notinmask/alt12/tarsgan_b8
# ./sched_l.sh 8 50 s5lw_notinmask/alt12/sis22_b4
# ./sched_l.sh 8 50 s5lw_notinmask/alt12/sis23_b1
# ./sched_l.sh 8 50 s5lw_notinmask/alt12/sis25_b8
# ./sched_l.sh 8 50 s5lw_notinmask/alt12/sis26_b8

# ./sched.sh 8 50 s5lw_notinmask/cur/dmnet
# ./sched.sh 8 50 s5lw_notinmask/cur/dsen2w
# ./sched.sh 8 50 s5lw_notinmask/cur/pix2pix
# ./sched.sh 8 50 s5lw_notinmask/cur/psgan
# ./sched.sh 8 50 s5lw_notinmask/cur/sis2
# ./sched.sh 8 50 s5lw_notinmask/cur/sis2-1
# ./sched.sh 8 50 s5lw_notinmask/cur/sis2-2
# ./sched.sh 8 50 s5lw_notinmask/cur/sis2-3
# ./sched.sh 8 50 s5lw_notinmask/cur/sis3
# ./sched.sh 8 50 s5lw_notinmask/cur/sis3-1
# ./sched.sh 8 50 s5lw_notinmask/cur/sis3-2
# ./sched.sh 8 50 s5lw_notinmask/cur/sis3-3
# ./sched.sh 8 50 s5lw_notinmask/cur/sis3-a
# ./sched.sh 8 50 s5lw_notinmask/cur/sis3-b
# ./sched.sh 8 50 s5lw_notinmask/cur/sis3-c
# ./sched.sh 8 50 s5lw_notinmask/cur/sis3-d
# ./sched.sh 8 50 s5lw_notinmask/cur/sis4
# ./sched.sh 8 50 s5lw_notinmask/cur/sis5
# ./sched.sh 8 50 s5lw_notinmask/cur/sis6
# ./sched.sh 8 50 s5lw_notinmask/cur/sis10
# ./sched.sh 8 50 s5lw_notinmask/cur/sis11
# ./sched.sh 8 50 s5lw_notinmask/cur/sis12
# ./sched.sh 8 50 s5lw_notinmask/cur/srgan
# ./sched.sh 8 50 s5lw_notinmask/cur/srs3
# ./sched_l.sh 8 50 s5lw_notinmask/cur/tarsgan_b8
# ./sched_l.sh 8 50 s5lw_notinmask/cur/sis22_b4
# ./sched_l.sh 8 50 s5lw_notinmask/cur/sis23_b1
# ./sched_l.sh 8 50 s5lw_notinmask/cur/sis25_b8
# ./sched_l.sh 8 50 s5lw_notinmask/cur/sis26_b8


### Restarting jobs, which were not finished after 8hrs (the initial config)

# ./sched.sh 8 50 s5lw_inmask/alt12/sis3 0804-1257
# ./sched.sh 8 50 s5lw_inmask/alt12/sis5 0807-0450
# ./sched.sh 8 50 s5lw_inmask/cur/pix2pix 0806-0413
# ./sched.sh 8 50 s5lw_inmask/cur/sis3-2 0805-0343
# ./sched_l.sh 8 50 s5lw_inmask/cur/sis3-3 0805-1916
# ./sched.sh 8 50 s5lw_inmask/cur/sis3-b 0805-0402
# ./sched.sh 8 50 s5lw_inmask/cur/srgan 0805-1956
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis3 0805-2050
# ./sched.sh 8 50 s5lw_notinmask/alt12/sis10 0806-1217
# ./sched.sh 8 50 s5lw_notinmask/cur/sis3-1 0808-0328
# ./sched_l.sh 8 50 s5lw_notinmask/cur/sis3-3 0805-1110
# ./sched.sh 8 50 s5lw_notinmask/cur/sis3-b 0805-1206
# ./sched.sh 8 50 s5lw_notinmask/cur/sis4 0805-1154
# ./sched.sh 8 50 s5lw_notinmask/cur/sis5 0806-0051

# ./sched.sh 8 50 s5lw_inmask/cur/sis3-c 0808-0912
# ./sched.sh 8 50 s5lw_notinmask/cur/sis3-d 0808-0953
# ./sched.sh 8 50 s5lw_inmask/alt12/sis6 0808-1150
# ./sched_l.sh 8 50 s5lw_notinmask/cur/sis6 0808-1516
# ./sched_l.sh 8 50 s5lw_inmask/alt12/sis25_b8 0808-1521
# ./sched_l.sh 8 50 s5lw_inmask/alt12/sis26_b8 0808-1521

# ./sched_l.sh 8 50 s5lw_inmask/cur/sis2 0808-1636
# ./sched_l.sh 8 50 s5lw_inmask/cur/sis3-a 0808-1720
# ./sched_l.sh 8 50 s5lw_inmask/cur/sis26_b8 0808-1758
# ./sched_l.sh 8 50 s5lw_notinmask/cur/sis5 0808-1758
# ./sched_l.sh 8 50 s5lw_inmask/cur/sis25_b8 0808-1758
# ./sched_l.sh 8 50 s5lw_notinmask/alt12/sis2-2 0808-1843
# ./sched_l.sh 8 50 s5lw_notinmask/alt12/sis25_b8 0808-2234
# ./sched_l.sh 8 50 s5lw_notinmask/alt12/sis26_b8 0808-2240
# ./sched_l.sh 8 50 s5lw_notinmask/cur/sis2-2 0808-2323
# ./sched_l.sh 8 50 s5lw_notinmask/cur/sis25_b8 0808-2329
# ./sched_l.sh 8 50 s5lw_notinmask/cur/sis26_b8 0808-2330
# ./sched_l.sh 8 50 s5lw_notinmask/alt12/sis3-3 0808-2346
# ./sched_l.sh 8 50 s5lw_inmask/alt12/sis22_b4 0808-2352

# ./sched_l.sh 8 50 s5lw_notinmask/cur/sis11 0809-0040
# ./sched_l.sh 8 50 s5lw_notinmask/cur/sis3-c 0809-0126
# ./sched_l.sh 8 50 s5lw_inmask/cur/sis22_b4 0809-0126


### Continutation to 100k steps

# ./sched_l.sh 12 50 s5lw_inmask/cur/sis2 0809-0915
# ./sched_l.sh 12 50 s5lw_inmask/cur/sis3 0809-1605
# ./sched_l.sh 12 50 s5lw_inmask/cur/sis4 0808-0839

# ./sched_l.sh 12 50 s5lw_inmask/alt12/sis2 0804-1611
# ./sched_l.sh 12 50 s5lw_inmask/alt12/sis3 0808-1516
# ./sched_l.sh 12 50 s5lw_inmask/alt12/sis4 0808-0924

# ./sched_l.sh 12 50 s5lw_notinmask/cur/sis2 0806-2218
# ./sched_l.sh 12 50 s5lw_notinmask/cur/sis3 0807-2346
# ./sched_l.sh 12 50 s5lw_notinmask/cur/sis4 0808-1710

# ./sched_l.sh 12 50 s5lw_notinmask/alt12/sis2 0808-1814
# ./sched_l.sh 12 50 s5lw_notinmask/alt12/sis3 0808-1516
# ./sched_l.sh 12 50 s5lw_notinmask/alt12/sis4 0809-0407
