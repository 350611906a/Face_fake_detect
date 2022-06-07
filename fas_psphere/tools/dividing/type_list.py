# date: 2022-01-07 17:28
# author: liucc
import os

raw_type_list = ['05', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '201', '202', '203', '204',
                 '205', '206', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'C1',
                 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
                 'MBLGMJliuxiaofei', 'MBLGTMpuyu', 'MCSSYTMzhaoyumiao', 'MGJMJlitingzhao', 'MGJTMbaochangchun',
                 'MGJTMqiuyanling', 'MNLMJzhangjianwu', 'MNLTMhuchangsheng', 'MPMMJcisunfei', 'MSYMJrenyuan',
                 'MSYSMJgaopeng', 'MSYSTMwangwenjie', 'MSYSgaopeng', 'MSZMJhuqidong', 'MSZMJmaorui', 'MSZMJmiaorentao',
                 'MSZMJwangweijie', 'MSZMJwangweijieA1E1E2', 'MSZMJyanmaochun', 'MSZTMqianxiaoli', 'MSZTMweiqiuhong',
                 'MSZTTqianxiaoli', 'MYPMTMhuaxuecheng', 'Myanmaochun', 'M0001', 'M0002', 'M0003', 'M0004', 'M0005',
                 'MGJMJ0002', 'MGJTT0002', 'MGJTT0003', 'MGJTT0005', 'MGJTT0010', 'MN10002', 'MN10003', 'MRJMJ0001',
                 'MRJMJ0003', 'MRJMJ0005', 'MRJMJ0006', 'MRJMJ0007', 'MRJMJ0009', 'MRJMJ0012', 'MRJMJ0019', 'MRJMJ0030',
                 'MRJMJ0031', 'MRJMJ0051', 'MRJMJ0052', 'MRJMJ0053', 'MRJMJwuzhongyou', 'MRJTF0034', 'MRJTMcaomengya',
                 'MRJTT0002', 'MRJTT0004', 'MRJTT0006', 'MRJTT0007', 'MRJTT0008', 'MRJTT0010', 'MRJTT0011', 'MRJTT0012',
                 'MRJTT0014', 'MRJTT0015', 'MRJTT0030', 'MRJTT0031', 'MRJTT0032', 'MRJTT0033', 'MRJTT0034',
                 'MRJTT00343',
                 'MRJTT0040', 'MRJTT0041', 'MRJTT0042', 'MRJTT0043', 'MRJTT0044', 'MRJTT0045', 'MRJTT0046', 'MRJTT0047',
                 'MRJTT0048', 'MRJTT0049', 'MRJTT0050', 'MRJTT0051', 'MRJTT0052', 'MRJZZ0004', 'MRJZZ0005', 'MRJZZ0006',
                 'MRTMJ0053', 'MSGMJ0002', 'MSLMJ0006', 'MSLMJ0008', 'MSLMJ0009', 'MSLMJ0010', 'MSLMJ0053', 'MSLMJ0054',
                 'MSLMJ0055', 'MSLMJ006', 'MSLMJ0200', 'MSLMJ0800', 'MSLMJ1700', 'MSLMJ2001', 'MSLMJ2002', 'MSLMJ2003',
                 'MSLTM0001', 'MSLTM0003', 'MSLTM009', 'PC1000', 'PC1045', 'PC4001', 'PC4002', 'PC4003', 'PC4004',
                 'PC4005', 'PC4006', 'PC4007', 'PC4008', 'PC4009', 'PC4010', 'PC4011', 'PC4012', 'PC4013', 'PC4014',
                 'PC4015', 'PC4016', 'PC4017', 'PC4018', 'PC4019', 'PC4020', 'PC4021', 'PC4022', 'PC4023', 'PC4024',
                 'PC4025', 'PC4026', 'PC4027', 'PC4028', 'PC4029', 'PC4030', 'PC4031', 'PC4032', 'PC4033', 'PC4034',
                 'PC4035', 'PC4036', 'PC4037', 'PC4038', 'PC4039', 'PCGT004', 'PCGT005', 'PCGT007', 'PCGT008',
                 'PCGT010', 'PCGT014', 'PCGT015', 'PCGT016', 'PCGT017', 'PCGT018', 'PCGT019', 'PCGT020', 'PCGT023',
                 'PCGT037', 'PCGT038', 'PCGT040', 'PCMJ0001', 'PCMJ0002', 'PCMJ0003', 'PCMJ0004', 'PCMJ0005',
                 'PCMJ0006',
                 'PCMJ0007', 'PCMJ0008', 'PCMJ0009', 'PCMJ0010', 'PCMJ0011', 'PCMJ0012', 'PCMJ0013', 'PCMJ006',
                 'PCMT0007',
                 'PCX2005', 'PCX2007', 'PCX2010', 'PCXZ0006', 'PCXZ001', 'PCXZ002', 'PCXZ003', 'PCXZ004', 'PCXZ005',
                 'PCXZ006', 'PCXZ007', 'PCXZ008', 'PCXZ009', 'PCXZ010', 'PCXZ011', 'PCXZ012', 'PCXZ013', 'PCXZ021',
                 'PCXZ022', 'PCXZ024', 'PCXZ025', 'PCXZ027', 'PCXZ028', 'PCXZ029', 'PCXZ030', 'PCXZ031', 'PCXZ032',
                 'PCXZ034', 'PCXZ041', 'PG4004', 'PG4005', 'PG4006', 'PG4007', 'PG5006', 'PGF01', 'PGF010', 'PGF0101',
                 'PGF011', 'PGF02', 'PGF021', 'PGF03', 'PGF031', 'PGF04', 'PGF041', 'PGF05', 'PGF051', 'PGF06',
                 'PGF061',
                 'PGF07', 'PGF071', 'PGF08', 'PGF081', 'PGF09', 'PGF091', 'PGO01', 'PGO02', 'PGO03', 'PGO04', 'PGO05',
                 'PGO06', 'PGO07', 'PGO08', 'PGO09', 'PGO10', 'PGT01', 'PGT02', 'PGT03', 'PGT04', 'PGT05', 'PGT06',
                 'PGT07', 'PGT08', 'PGT09', 'PGT10', 'PH0004', 'PH0005', 'PH0006', 'PH0007', 'PH1802', 'PH8001',
                 'PH8002',
                 'PH8003', 'PH8004', 'PH8006', 'PH8008', 'PH8010', 'PH8014', 'PI0001', 'PI0002', 'PI0003', 'PI0004',
                 'PI0005', 'PI0006', 'PI0007', 'PI0008', 'PI0009', 'PI0010', 'PI0011', 'PI0012', 'PI0013', 'PI0014',
                 'PI0015', 'PI0016', 'PI0017', 'PI0018', 'PI0019', 'PI0020', 'PI0021', 'PI0022', 'PI0023', 'PI0024',
                 'PI0025', 'PI0026', 'PI0027', 'PI0028', 'PI0029', 'PI0030', 'PI0031', 'PLS01', 'PLS02', 'PLS03',
                 'PLS04', 'PLS05', 'PLS06', 'PLS07', 'PLS08', 'PLS09', 'PLS10', 'PLS11', 'PLS12', 'PLS13', 'PLS14',
                 'PLS15', 'PLS16', 'PLS17', 'PLS18', 'PLS19', 'PLS20', 'PMSZP001', 'PMSZP001A1', 'PQ1001', 'PQ1002',
                 'PQ1003', 'PQ1004', 'PQ1005', 'PQ1006', 'PQ1007', 'PQ1008', 'PQ1009', 'PQ1010', 'PQ1011', 'PQ1012',
                 'PQ1013', 'PQ1014', 'PQ1015', 'PQ1017', 'PQ1018', 'PQ1019', 'PQ1020', 'PQ2001', 'PQ2002', 'PQ2003',
                 'PQ2004', 'PQ2005', 'PQ2006', 'PQ2007', 'PQ2008', 'PQ2009', 'PQ2010', 'PQ2011', 'PQ2012', 'PQ2013',
                 'PQ2014', 'PQ2015', 'PQ2016', 'PQ2017', 'PQ2019', 'PQ2020', 'PQ2022', 'PQ2024', 'PQ3001', 'PQ3002',
                 'PQ3003', 'PQ3004', 'PQ3005', 'PQ3006', 'PQ3007', 'PQ3008', 'PQ3009', 'PQ3010', 'PQ3011', 'PQ3012',
                 'PQ3013', 'PQ3014', 'PQ3015', 'PQ3016', 'PQ3017', 'PQ3018', 'PQ3021', 'PQ3023', 'PQ6006', 'PQ6007',
                 'PQ6119', 'PQ6120', 'PQ6121', 'PQ6122', 'PQ6123', 'PQ6124', 'PQ6125', 'PQ6131', 'PQ6132', 'PQ6133',
                 'PQ6134', 'PQ6137', 'PRPT0001', 'PRPT0002', 'PRPT0003', 'PRPT0004', 'PRPT0005', 'PRPT0006', 'PRPT0007',
                 'PRPT0008', 'PRPT0009', 'PRPT0010', 'PRPT0011', 'PRPT0012', 'PRPT0013', 'PRPT0014', 'PRPT0015',
                 'PRPT0016', 'PRPT0017', 'PRPT0018', 'PRPT0019', 'PRPT0020', 'PRPT0021', 'PRPT015', 'PW4001', 'PW4002',
                 'PW4003', 'PW4004', 'PW4005', 'PW4006', 'PW4007', 'PX0002', 'PX0003', 'PX0004', 'PX0005', 'PX0006',
                 'PX0007', 'PX0009', 'PX0010', 'PX0011', 'PX0012', 'PX0013', 'PX0014', 'PX0015', 'PX0016', 'PX0017',
                 'PX0018', 'PX0019', 'PX0020', 'PX0021', 'PX0022', 'PX0024', 'PX0025', 'PX0026', 'PX0027', 'PX0028',
                 'PX0029', 'PX0033', 'Pbaocc', 'Pchenzc', 'Pcuizhe', 'Pdaicl', 'Pfancp', 'Pfuxq', 'Pganjian', 'Phewu',
                 'Phuaxc', 'Phucs', 'Phuyz', 'Phuzy', 'Pi0012', 'Pjiangkun', 'Pkangkai', 'Plianghan', 'Plianjie',
                 'Plitz',
                 'Pmaorui', 'Ppuyu', 'Psunak', 'Pwangyy', 'Pweimeng', 'Pxueyuan', 'Pxuxk', 'Pxx', 'Pzb', 'Pzhangjs',
                 'Pzhaoxd', 'Pzjz', 'Pzw', 'Pzyy', 'W0001', 'W0002', 'W0003', 'W0004', 'W0005'
                 ]


def get_type_set(raw_list):
    type_3d_list, type_2d_list = [], []
    for rl in raw_list:
        e_str = ''
        for e in rl:
            if not e.isupper():
                break
            e_str += e
        if len(e_str) == 0:
            continue
        if e_str[0] == "P":
            type_2d_list.append(e_str + ("-D" if rl[len(e_str)].isdigit() else "-L"))
        elif len(e_str) == len(rl):
            type_3d_list.append(e_str + "-N")
        else:
            type_3d_list.append(e_str + ("-D" if rl[len(e_str)].isdigit() else "-L"))
    type_set_3d, type_set_2d = set(type_3d_list), set(type_2d_list)
    return type_set_3d, type_set_2d


def fake_samples_type_set(data_set_dir_root=None):
    if data_set_dir_root is not None:
        rtl = []
        for dir_path, dir_names, file_names in os.walk(data_set_dir_root):
            for dn in dir_names:
                rtl.append(dn.split("_")[0])
    else:
        rtl = raw_type_list
    return get_type_set(rtl)


if __name__ == "__main__":
    ts_3d, ts_2d = fake_samples_type_set(
        data_set_dir_root='/storage-server5/dataWarehouse/lvmi/data/fake/'
    )
    print('3d(%d):' % len(ts_3d), ts_3d)
    print('2d(%d):' % len(ts_2d), ts_2d)






















