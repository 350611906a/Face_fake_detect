'''
功能：
    对结果进行分析，主要可实现如下功能，
    1、对输出结果的list进行处理，将错误的case全部输出，形成新的list；
    2、将分类错误的图片，保存到指定位置，按照文件夹名字保存，并包含前一级路径；
    3、读取拷贝的图片，并将计算得分和阈值打印在图片上，并添加后缀[_result]保存;
新增功能：
    1、将关键点绘制在结果图片上；
        print_key_point 为功能开关，0表示不画关键点，1表示画关键点
        

作者：
    王文杰
'''

import os
import shutil
import cv2
print_key_point = 1

# result_list = "/workspace/wangwj/dilusense_work/DM8/fas_test/checkpoints/dm8/IFAS_dm8_v1.3/epoch-272_lvmi_20220301_ir_M_all.txt"
# result_list = "/workspace/wangwj/dilusense_work/DM8/fas_test/checkpoints/dm8/IFAS_dm8_v1.3/epoch-272_IFAS_lvmi_FAS2022_M_test.txt"
# result_list = "/workspace/wangwj/dilusense_work/DM8/fas_test/checkpoints/dm8/IFAS_dm8_v1.3/model_pth/epoch_28x/epoch-285_lv_ir_live_faceinfo_0.88_test_G_all_noocc.txt"
# result_list = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/test_old_train_checkpoint/epoch_26x/epoch-267_fake_20220410_G_lvmi_faceinfo_epoch-117_352_320_0.88.txt'
# result_list = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/test_checkpoint_test1/epoch-266_lv_20220312_ir_live_faceinfo_epoch-117_352_320_0.88_test_G_kids_all_noocc.txt'
# result_list = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/test_checkpoint_test3/epoch_270_279/epoch-275_fake_20220410_G_lvmi_faceinfo_epoch-117_352_320_0.88.txt'
result_list = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/test_checkpoint_test3/epoch_280_287/epoch-282_fake_20220410_G_lvmi_faceinfo_epoch-117_352_320_0.88.txt'
result_list = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/test_checkpoint_test3/epoch_280_287/epoch-282_DM8003_20220413_ir_live_faceinfo_epoch-117_352_320_0.88_test_G_.txt'
result_list = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/test_checkpoint_test3/epoch_280_287/epoch-282_DM8003_20220418_ir_live_faceinfo_epoch-117_352_320_0.88_test_G_.txt'
result_list = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/test_checkpoint_test4/epoch_290_299/epoch-297_fake_20220410_G_lvmi_faceinfo_epoch-117_352_320_0.88.txt'
result_list = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/test_checkpoint_test4/epoch_283-289/epoch-302_fake_20220410_G_lvmi_faceinfo_epoch-117_352_320_0.88.txt'
result_list = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/test_checkpoint_test4/epoch_283-289/epoch-300_fake_20220410_G_lvmi_faceinfo_epoch-117_352_320_0.88.txt'
result_list = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/test_checkpoint_test4/epoch_303_315/epoch-312_fake_20220410_G_lvmi_faceinfo_epoch-117_352_320_0.88.txt'
result_list = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/test_checkpoint_test4/epoch_283-289/epoch-300_IR_20220423_G_all_faceinfo_epoch-117_352_320_0.88.txt'
result_list = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/test_checkpoint_test4/epoch_303_315/epoch-314_IR_20220423_G_all_faceinfo_epoch-117_352_320_0.88.txt'
result_list = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/test_checkpoint_test4/epoch_283-289/epoch-300_BCTC_LSZ_20220424_M_all_faceinfo_epoch-117_352_320_0.88.txt'
result_list = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/test_checkpoint_test4/epoch_283-289/epoch-300_BCTC_hezongxianzong_G_all_faceinfo_epoch-117_352_320_0.88.txt'
result_list = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/test_checkpoint_test5/bosong_ronghe/weights/epoch-315_BCTC_LSZ_20220424_M_all_faceinfo_epoch-117_352_320_0.88.txt'
result_list = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/test_checkpoint_test6/epoch-319_BCTC_hezongxianzong_G_all_faceinfo_epoch-117_352_320_0.88.txt'
result_list = '/ssd1/wangwj/work_dilusense/DM8/fas_test_1/checkpoints/dm8/xianzong/epoch-230_IFAS_lvmi_FAS2022_M_test.txt'
result_list = '/ssd1/wangwj/work_dilusense/RGBD/fas/pre_checkpoint/epoch-45_IFAS_rgbd_20211227_G_test.txt'
result_list = '/ssd1/wangwj/work_dilusense/RGBD/fas/pre_checkpoint/epoch-52_IFAS_rgbd_20211111_G_test.txt'
result_list = '/ssd1/wangwj/work_dilusense/RGBD/fas/pre_checkpoint/epoch-78_IFAS_rgbd_20211111_G_test.txt'
result_list = '/ssd1/wangwj/work_dilusense/RGBD/fas/pre_checkpoint/epoch-78_IFAS_rgbd_20211227_G_test.txt'
result_list = '/ssd1/wangwj/work_dilusense/RGBD/fas/pre_checkpoint/epoch-52_IFAS_rgbd_20211111_M_test.txt'
result_list = '/ssd1/wangwj/work_dilusense/RGBD/fas/pre_checkpoint/epoch-78_ir_RGBD_faceinfo_epoch-117_352_320_0.88_G_RGBD.txt'
result_list = '/ssd1/wangwj/work_dilusense/RGBD/fas/pre_checkpoint/epoch-78_ir_RGBD_faceinfo_epoch-117_352_320_0.88_G_RGBD_tongyikeypoint.txt'
result_list = '/ssd1/wangwj/work_dilusense/RGBD/fas/tmp/lvmi_mobileNet_autoRadius_k0_test4_solver_tain_epoch104/weights/new/epoch-223_ir_RGBD_faceinfo_epoch-117_352_320_0.88_G_RGBD_tongyikeypoint.txt'

# result_list = '/ssd1/wangwj/work_dilusense/RGBD/fas/pre_checkpoint/epoch-78_ir_RGBD_faceinfo_epoch-117_352_320_0.88_G_RGBD_tongyikeypoint.txt'
result_list = '/ssd1/wangwj/work_dilusense/RGBD/fas/pre_checkpoint/epoch-78_ir_RGBD_faceinfo_epoch-117_352_320_0.88_G_RGBD.txt'

if print_key_point == 1:
    # 人脸信息的list
    # faceinfo_list_file = '/ssd1/wangwj/work_dilusense/RGBD/fas/dataset/ir_RGBD/list/ir_RGBD_faceinfo_epoch-117_352_320_0.88_G_RGBD_tongyikeypoint.txt'
    faceinfo_list_file = '/ssd1/wangwj/work_dilusense/RGBD/fas/dataset/ir_RGBD/list/ir_RGBD_faceinfo_epoch-117_352_320_0.88_G_RGBD.txt'

def run_analysis(fp_list):
    # 生成路径
    tmp = result_list.split('/')[-1]
    path = result_list.strip(tmp)
    path = path + "result_analysis_test20220523/" + tmp.strip('.txt')
    print(path)

    if not os.path.exists(path):
        os.makedirs(path)

    error_list = path + '/' + 'error_list.txt'
    fp_dst = open(error_list, 'w')

    for line in fp_list:
        # if '0,/' in line:
        if True:
            fp_dst.write(line)

            img_path = line.split(',')[1]
            img_path_dst = img_path.split('/')[-2]
            img_path_dst = path + '/' + img_path_dst
            if not os.path.exists(img_path_dst):
                os.makedirs(img_path_dst)
            shutil.copy(img_path, img_path_dst)

            # 保存文件
            ## 获取新的文件名
            tmp_img_name = line.split(',')[1].split('/')[-1]
            new_img_path = tmp_img_name.replace('.png', '_result.png')
            new_img_path = img_path_dst + '/' + new_img_path
            
            ## 获取阈值及得分
            score_thd = line.split(',')[3]
            score = line.split(',')[2]

            ## 读取图片并写入阈值及得分
            img = cv2.imread(img_path)
            # cv2.putText(img, 'thd: ' + str(score_thd), (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (240, 0, 0), 4)
            # cv2.putText(img, 'score: ' + str(score), (20, 120), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 240), 4)
            cv2.putText(img, 'thd: ' + str(score_thd), (20, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (240, 0, 0), 1)
            cv2.putText(img, 'score: ' + str(score), (20, 120), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 240), 1)

            if print_key_point == 1:
                fp_face_info = open(faceinfo_list_file, 'r')
                ## 获取人脸的关键点信息
                key_point = []
                for line_faceinfo in fp_face_info:
                    if img_path in line_faceinfo:
                        tmp_key_point = line_faceinfo.split(';')[-1].split(' ')[0]
                        key_point = tmp_key_point.split(',')
                        break
                fp_face_info.close()
                assert(len(key_point) != 0)
                # print(key_point)

                key_point_int = []
                for i in key_point:
                    key_point_int.append(int(float(i)))
                # print(key_point_int)

                ## 在关键点位置画圆
                cv2.circle(img, (key_point_int[0], key_point_int[1]), 1, (0,0,255), -1, 4, 0)
                cv2.circle(img, (key_point_int[2], key_point_int[3]), 1, (0,0,255), -1, 4, 0)
                cv2.circle(img, (key_point_int[4], key_point_int[5]), 1, (0,0,255), -1, 4, 0)
                cv2.circle(img, (key_point_int[6], key_point_int[7]), 1, (0,0,255), -1, 4, 0)
                cv2.circle(img, (key_point_int[8], key_point_int[9]), 1, (0,0,255), -1, 4, 0)

            ## 保存图片
            cv2.imwrite(new_img_path, img)

    fp_dst.close()



if __name__ == "__main__":
    with open(result_list, 'r')as fp_list:
        run_analysis(fp_list)
