from all_data_preprocess.dataloader import *
from model.main_models import *
import torch.optim as optim
from all_loss.encode_loss import *
from evaluation import Evaluation
from skimage.filters.thresholding import threshold_otsu
from PIL import Image
import time
from thop import profile, clever_format
import torchvision.transforms as transforms
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")






def initialize_model(image):

    if len(image.shape)==3:
        encoder = GK_Encode(g_in_features=3, g_hidden=3, g_out_features=2, k_in_features=3, k_hidden=3,
                            k_out_features=2, dropout=0.1).to(device)
        decoder_t1 = ComprehensiveDecoder(g_in_features=2, g_hidden=3, g_out_features=3, k_in_features=2, k_hidden=3,
                                          k_out_features=3, dropout=0.1).to(device)
        decoder_t2 = ComprehensiveDecoder(g_in_features=2, g_hidden=3, g_out_features=3, k_in_features=2, k_hidden=3,
                                          k_out_features=3, dropout=0.1).to(device)
    else:
        encoder = GK_Encode(g_in_features=1, g_hidden=1, g_out_features=1, k_in_features=1, k_hidden=1,
                            k_out_features=1, dropout=0.1).to(device)
        decoder_t1 = ComprehensiveDecoder(g_in_features=1, g_hidden=1, g_out_features=1, k_in_features=1,
                                          k_hidden=1, k_out_features=1, dropout=0.1).to(device)
        decoder_t2 = ComprehensiveDecoder(g_in_features=1, g_hidden=1, g_out_features=1, k_in_features=1,
                                          k_hidden=1, k_out_features=1, dropout=0.1).to(device)

    optimizer_e_t1 = optim.Adam(encoder.parameters(), lr=0.0001,weight_decay=0.0001 )

    optimizer_d_t1 = optim.Adam(decoder_t1.parameters(), lr=0.0001, weight_decay=0.0001)
    optimizer_d_t2 = optim.Adam(decoder_t2.parameters(), lr=0.0001,weight_decay=0.0001)
    return {'encoder':encoder,
        'decoder_t1':decoder_t1,
        'decoder_t2': decoder_t2,
        'optimizer_e_t1':optimizer_e_t1,
        'optimizer_d_t1':optimizer_d_t1,
        'optimizer_d_t2': optimizer_d_t2

        }


def train(epoch,graph,model,save_path):
    model['encoder'].train()
    model['decoder_t1'].train()
    model['decoder_t2'].train()

    batch_size = 10
    total_loss = 0


    # 计算总批次数
    num_batches = (graph['obj_nums'] + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        # 清零梯度
        model['optimizer_e_t1'].zero_grad()
        model['optimizer_d_t1'].zero_grad()
        model['optimizer_d_t2'].zero_grad()

        batch_loss = 0

        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, graph['obj_nums'])

        for _iter in range(batch_start, batch_end):
            node_t1 = graph['all_Superpixel_node_t1'][_iter].to(device)
            f_adj_t1 = graph['feature_graphs_t1'][_iter].to(device)
            s_adj_t1 = graph['structure_graphs_t1'][_iter].to(device).type(torch.float32)

            node_t2 = graph['all_Superpixel_node_t2'][_iter].to(device)
            f_adj_t2 = graph['feature_graphs_t2'][_iter].to(device)
            s_adj_t2 = graph['structure_graphs_t2'][_iter].to(device).type(torch.float32)

            # 编码t1
            (output_f_t1,  output_s_t1,  output_t1) = model['encoder'](node_t1,f_adj_t1,s_adj_t1)
            # 编码t2
            (output_f_t2,  output_s_t2,  output_t2) = model['encoder'](node_t2,f_adj_t2,s_adj_t2)
            # 解码
            (output_rec_f_t1, output_rec_s_t1) = model['decoder_t1'](
                output_t1['pooled_output'], f_adj_t1, s_adj_t1)
            (output_rec_f_t2, output_rec_s_t2) = model['decoder_t2'](
                output_t2['pooled_output'], f_adj_t2, s_adj_t2)
            #共性损失
            loss_com_t2 = common_loss(output_t1['pooled_output'], output_t2['pooled_output'])#通道注意力融合之后


            # 计算重构损失
            rec_s_t1_gcn_loss = F.mse_loss(output_rec_s_t1['gk'], node_t1)
            rec_f_t1_gcn_loss = F.mse_loss(output_rec_f_t1['gk'], node_t1)

            rec_s_t2_gcn_loss = F.mse_loss(output_rec_s_t2['gk'], node_t2)
            rec_f_t2_gcn_loss = F.mse_loss(output_rec_f_t2['gk'], node_t2)


            # 计算总损失
            loss_t1_total = (rec_s_t1_gcn_loss + rec_f_t1_gcn_loss
                             )
            loss_t2_total = (rec_s_t2_gcn_loss + rec_f_t2_gcn_loss
                             )

            loss = loss_t1_total + loss_t2_total + loss_com_t2

            # 累积批次损失
            batch_loss = batch_loss+loss


        # 计算批次平均损失并反向传播
        batch_loss = batch_loss / (batch_end - batch_start)

        batch_loss.backward()

        # 更新参数
        model['optimizer_e_t1'].step()
        model['optimizer_d_t1'].step()
        model['optimizer_d_t2'].step()

        total_loss += batch_loss.item()
    weight_path = os.path.join(save_path, 'weight_epoch_%d.pth' % (epoch))
    model_dict = {
        'encoder': model['encoder'].state_dict(),
        'decoder_a': model['decoder_t1'].state_dict(),
        'decoder_b': model['decoder_t2'].state_dict()
    }
    torch.save(model_dict, weight_path)

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch}, Loss: {avg_loss}")


def test(epoch, model, graph, image_t1,ref, objects, save_path, N_SEG, Com):
    model['encoder'].eval()
    model['decoder_t1'].eval()
    model['decoder_t2'].eval()
    with torch.no_grad():
        diff_set = []
        for _iter in range(graph['obj_nums']):
            node_t1 = graph['all_Superpixel_node_t1'][_iter].to(device)
            f_adj_t1 = graph['feature_graphs_t1'][_iter].to(device)
            s_adj_t1 = graph['structure_graphs_t1'][_iter].to(device).type(torch.float32)

            node_t2 = graph['all_Superpixel_node_t2'][_iter].to(device)
            f_adj_t2 = graph['feature_graphs_t2'][_iter].to(device)
            s_adj_t2 = graph['structure_graphs_t2'][_iter].to(device).type(torch.float32)

            # 编码t1
            (output_f_t1, output_s_t1, output_t1) = model['encoder'](node_t1, f_adj_t1, s_adj_t1)
            # 编码t2
            (output_f_t2, output_s_t2, output_t2) = model['encoder'](node_t2, f_adj_t2, s_adj_t2)

            difference_map_x = F.mse_loss(output_t1['pooled_output'], output_t2['pooled_output'])
            diff_set.append(difference_map_x)

        if ref.shape[-1] == 3:
            diff_map = torch.zeros((image_t1.shape[0], image_t1.shape[1], image_t1.shape[-1]), device=device)
        else:
            diff_map = torch.zeros((image_t1.shape[0], image_t1.shape[1]), device=device)
        for i in range(0, graph['obj_nums'] ):
            diff_map[objects == i] = diff_set[i]

        diff_map_min = diff_map.min()

        diff_map_max = diff_map.max()
        if diff_map_max - diff_map_min != 0:
            diff_map_nor = (diff_map - diff_map_min) / (diff_map_max - diff_map_min)
        else:
            small_constant = 1e-10
            diff_map_nor = (diff_map - diff_map_min) / (diff_map_max - diff_map_min + small_constant)

        filename1 = f"DI_iter_{epoch}.png"
        filename2 = f"CM_iter_{epoch}.png"
        diff_map_save = diff_map_nor * 255
        diff_map_save = diff_map_save.cpu().numpy()
        if image_t1.shape[0]==800:
            numpy_array = diff_map_nor.cpu().numpy()
            numpy_array = (numpy_array * 255).astype(np.uint8)

            diff_map_save_pil = Image.fromarray(numpy_array)
            transform = transforms.Compose([
                transforms.Resize((2000, 2000)),  # 将图像缩小到指定的尺寸
            ])

            # 应用 transform
            diff_map_save_size = transform(diff_map_save_pil)
            diff_map_save = np.array(diff_map_save_size)

        io.imsave(os.path.join(save_path, filename1), diff_map_save.astype(np.uint8))
        thre1 = threshold_otsu(diff_map_save)
        CM1 = (diff_map_save >= thre1) * 255
        io.imsave(os.path.join(save_path, filename2), (CM1).astype(np.uint8))
        Indicators1 = Evaluation(ref, CM1)
        OA1, kappa1, AA1 = Indicators1.Classification_indicators()
        P1, R1, F11 = Indicators1.ObjectExtract_indicators()
        TP1, TN1, FP1, FN1 = Indicators1.matrix()

        file_name = "eval.txt"
        # # 将内容写入文件
        file_path_txt = os.path.join(save_path, file_name)
        val_acc = open(file_path_txt, 'a')
        val_acc.write(
            '===============================Parameters settings==============================\n')
        val_acc.write('=== epoch={} || superpixel Num={} || compact ={} ===\n'.format(epoch, N_SEG, Com))
        val_acc.write('Domain t1:\n')
        val_acc.write('TP={} || TN={} || FP={} || FN={}\n'.format(TP1, TN1, FP1, FN1))
        val_acc.write("\"OA\":\"" + "{}\"\n".format(OA1))
        val_acc.write("\"Kappa\":\"" + "{}\"\n".format(kappa1))
        val_acc.write("\"AA\":\"" + "{}\"\n".format(AA1))
        val_acc.write("\"Precision\":\"" + "{}\"\n".format(P1))
        val_acc.write("\"Recall\":\"" + "{}\"\n".format(R1))
        val_acc.write("\"F1\":\"" + "{}\"\n".format(F11))
        val_acc.close()
def main(data_name,epoch):

    if data_name=='Italy':
        for N_SEG in [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]:
            for Com in [20,22,24,26,28,30,32,34,36,38,40]:
                # 加载数据
                graph, image_t1, ref, objects, di_sp_b = load_images_Italy(N_SEG=N_SEG, Com=Com)

                for item in range(10):
                    # 生成特定 item 的文件夹路径
                    save_path = f"all_result/Italy/di_sp_{N_SEG}_{Com}_{item}"
                    os.makedirs(save_path, exist_ok=True)

                    # 保存 di_sp_b 图像
                    io.imsave(os.path.join(save_path, f'di_sp_slic_{N_SEG}_{Com}.bmp'),
                              (di_sp_b * 255).astype(np.uint8))

                    # 初始化模型
                    model = initialize_model(image_t1)
                    # 在该轮训练中进行 epoch 次训练
                    for i in range(epoch):
                        train(i, graph=graph, model=model, save_path=save_path)
                        test(i, model, graph, image_t1, ref, objects, save_path, N_SEG, Com)

                    # 清除模型，释放显存
                    del model
                    torch.cuda.empty_cache()

                # 清除加载的数据
                del graph, image_t1, ref, objects
                torch.cuda.empty_cache()
    elif data_name== 'yellow':
        for N_SEG in [2400]:
            for Com in [0.30]:
                # 加载数据
                graph, image_t1, ref, objects, di_sp_b = load_images_yellow(N_SEG=N_SEG, Com=Com)

                for item in range(10):
                    # 生成特定 item 的文件夹路径
                    save_path = f"all_result/yellow/di_sp_{N_SEG}_{Com}_{item}"
                    os.makedirs(save_path, exist_ok=True)

                    # 保存 di_sp_b 图像
                    io.imsave(os.path.join(save_path, f'di_sp_slic_{N_SEG}_{Com}.bmp'),
                              (di_sp_b * 255).astype(np.uint8))

                    # 初始化模型
                    model = initialize_model(image_t1)

                    # 在该轮训练中进行 epoch 次训练
                    for i in range(epoch):
                        train(i, graph=graph, model=model, save_path=save_path)
                        test(i, model, graph, image_t1, ref, objects, save_path, N_SEG, Com)

                    # 清除模型，释放显存
                    del model
                    torch.cuda.empty_cache()

                # 清除加载的数据
                del graph, image_t1, ref, objects
                torch.cuda.empty_cache()
    elif data_name== 'dawn':
        for N_SEG in [1200]:
            for Com in [24]:
                # 加载数据
                graph, image_t1, ref, objects, di_sp_b = load_images_Dawn(N_SEG=N_SEG, Com=Com)

                for item in range(5):

                    # 生成特定 item 的文件夹路径
                    save_path = f"all_result/dawn/di_sp_{N_SEG}_{Com}_{item}"
                    os.makedirs(save_path, exist_ok=True)

                    # 保存 di_sp_b 图像
                    io.imsave(os.path.join(save_path, f'di_sp_slic_{N_SEG}_{Com}.bmp'),
                              (di_sp_b * 255).astype(np.uint8))

                    # 初始化模型
                    model = initialize_model(image_t1)

                    # 在该轮训练中进行 epoch 次训练
                    for i in range(epoch):
                        train(i, graph=graph, model=model, save_path=save_path)
                        test(i, model, graph, image_t1, ref, objects, save_path, N_SEG, Com)

                    # 清除模型，释放显存
                    del model
                    torch.cuda.empty_cache()

                # 清除加载的数据
                del graph, image_t1, ref, objects
                torch.cuda.empty_cache()

    elif data_name== 'gloucester2':
        for N_SEG in [1600]:
            for Com in [20]:
                # 加载数据
                graph, image_t1, ref, objects, di_sp_b = load_images_Gloucester2(N_SEG=N_SEG, Com=Com)

                for item in range(5):
                    # 生成特定 item 的文件夹路径
                    save_path = f"all_result/gloucester2/di_sp_{N_SEG}_{Com}_{item}"
                    os.makedirs(save_path, exist_ok=True)

                    # 保存 di_sp_b 图像
                    io.imsave(os.path.join(save_path, f'di_sp_slic_{N_SEG}_{Com}.bmp'),
                              (di_sp_b * 255).astype(np.uint8))

                    # 初始化模型
                    model = initialize_model(image_t1)

                    # 在该轮训练中进行 epoch 次训练
                    for i in range(epoch):
                        train(i, graph=graph, model=model, save_path=save_path)
                        test(i, model, graph, image_t1, ref, objects, save_path, N_SEG, Com)

                    # 清除模型，释放显存
                    del model
                    torch.cuda.empty_cache()

                # 清除加载的数据
                del graph, image_t1, ref, objects
                torch.cuda.empty_cache()
    elif data_name== 'gloucester1':
        for N_SEG in [1000]:
            for Com in [30]:
                # 加载数据
                graph, image_t1, ref, objects, di_sp_b = load_images_Gloucester1(N_SEG=N_SEG, Com=Com)

                for item in range(5):
                    # 生成特定 item 的文件夹路径
                    save_path = f"all_result/gloucester1/di_sp_{N_SEG}_{Com}_{item}"
                    os.makedirs(save_path, exist_ok=True)

                    # 保存 di_sp_b 图像
                    io.imsave(os.path.join(save_path, f'di_sp_slic_{N_SEG}_{Com}.bmp'),
                              (di_sp_b * 255).astype(np.uint8))

                    # 初始化模型
                    model = initialize_model(image_t1)

                    # 在该轮训练中进行 epoch 次训练
                    for i in range(epoch):
                        train(i, graph=graph, model=model, save_path=save_path)
                        test(i, model, graph, image_t1, ref, objects, save_path, N_SEG, Com)
                    # 清除模型，释放显存
                    del model
                    torch.cuda.empty_cache()

                    # 清除加载的数据
                del graph, image_t1, ref, objects
                torch.cuda.empty_cache()


    elif data_name== 'california':
        for N_SEG in [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]:
            for Com in [0.2,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38,0.40]:
                # 加载数据
                graph, image_t1, ref, objects, di_sp_b = load_images_California(N_SEG=N_SEG, Com=Com)

                for item in range(10):
                    # 生成特定 item 的文件夹路径
                    save_path = f"all_result/california/di_sp_{N_SEG}_{Com}_{item}"
                    os.makedirs(save_path, exist_ok=True)

                    # 保存 di_sp_b 图像
                    io.imsave(os.path.join(save_path, f'di_sp_slic_{N_SEG}_{Com}.bmp'),
                              (di_sp_b * 255).astype(np.uint8))

                    # 初始化模型
                    model = initialize_model(image_t1)

                    # 在该轮训练中进行 epoch 次训练
                    for i in range(epoch):
                        train(i, graph=graph, model=model, save_path=save_path)
                        test(i, model, graph, image_t1, ref, objects, save_path, N_SEG, Com)

                    # 清除模型，释放显存
                    del model
                    torch.cuda.empty_cache()

                # 清除加载的数据
                del graph, image_t1, ref, objects
                torch.cuda.empty_cache()
    elif data_name == 'france':
        for N_SEG in [1000]:
            for Com in [10]:
                # 加载数据
                graph, image_t1, ref, objects, di_sp_b = load_images_France(N_SEG=N_SEG, Com=Com)

                for item in range(3):
                    # 生成特定 item 的文件夹路径
                    save_path = f"all_result/France/di_sp_{N_SEG}_{Com}_{item}"
                    os.makedirs(save_path, exist_ok=True)

                    # 保存 di_sp_b 图像
                    io.imsave(os.path.join(save_path, f'di_sp_slic_{N_SEG}_{Com}.bmp'),
                              (di_sp_b * 255).astype(np.uint8))

                    # 初始化模型
                    model = initialize_model(image_t1)
                    encoder = model['encoder']

                    # 保存结果
                    with open(os.path.join(save_path, "flops_runtime.txt"), "w") as f:
                        f.write(f"FLOPs: {flops}\n")
                        f.write(f"Params: {params}\n")
                        f.write(f"Avg Inference Time per superpixel (s): {avg_time:.6f}\n")

                    print(f"[INFO] FLOPs={flops}, Params={params}, Avg Time per superpixel={avg_time:.6f}s")

                    # 在该轮训练中进行 epoch 次训练
                    for i in range(epoch):
                        train(i, graph=graph, model=model, save_path=save_path)
                        test(i, model, graph, image_t1, ref, objects, save_path, N_SEG, Com)

                    # 清除模型，释放显存
                    del model
                    torch.cuda.empty_cache()

                # 清除加载的数据
                del graph, image_t1, ref, objects






                torch.cuda.empty_cache()

if __name__ == '__main__':
    for i in ['dawn','france','gloucester1','gloucester2','yellow']:
        main(data_name=i,epoch=30)