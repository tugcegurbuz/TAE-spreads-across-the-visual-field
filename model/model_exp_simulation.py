import model
import torch

class runSimulation:
    def __init__(self):
        super().__init__()
    def model_response(batch_size, p_no, inputs, condition):
        theta = inputs[:, 2].view(batch_size, 1)
        xs= inputs[:, 0].view(batch_size, 1, 1)
        ys = inputs[:, 1].view(batch_size, 1, 1)

        if condition == 'adap':
            model = adapModel()
            trained_model_dir = '<>' + str(p) + '__checkpoint_adap_model.pth.tar'
            model.load_state_dict(torch.load(trained_model_dir))
            A_a, rho_hat = model(batch_size, theta, xs, ys)
        else:
            model = noAdapModel()
            working_folder = '<Enter dir>'
            filedir = working_folder + '<>' + str(p_no) + '_params_all.pkl'
            with open(filedir, 'rb') as f:
                data = pickle.load(f)
            sigma_x = data[0][4][0]
            sigma_y = data[0][5][0]
            A, rho_hat = model(batch_size, theta, xs, ys, sigma_x, sigma_y)
        return rho_hat
    def experiment(batch_size, trained_model_dir, p_no, condition):
        x_sti = []
        y_sti = []
        raw_test_locs = []
        perceived_tilt = []
        orientations = []
        start_angle_sc1 = 105
        start_angle_sc2 = 80
        sc_states = [
                [[2.5, 6, 1, 'dN2'], [start_angle_sc1], [], 0],
                [[2.5, 6, 2, 'dN2'], [start_angle_sc2], [], 0],
                [[6.5, 6, 1, 'dN1'], [start_angle_sc1], [], 0],
                [[6.5, 6, 2, 'dN1'], [start_angle_sc2], [], 0],
                [[10.5, 6, 1, 'dC'], [start_angle_sc1], [], 0],
                [[10.5, 6, 2, 'dC'], [start_angle_sc2], [], 0],
                [[14.5, 6, 1, 'dF1'], [start_angle_sc1], [], 0],
                [[14.5, 6, 2, 'dF1'], [start_angle_sc2], [], 0],
                [[18.5, 6, 1, 'dF2'], [start_angle_sc1], [], 0],
                [[18.5, 6, 2, 'dF2'], [start_angle_sc2], [], 0],
                [[2.5, 10, 1, 'cN2'], [start_angle_sc1], [], 0],
                [[2.5, 10, 2, 'cN2'], [start_angle_sc2], [], 0],
                [[6.5, 10, 1, 'cN1'], [start_angle_sc1], [], 0],
                [[6.5, 10, 2, 'cN1'], [start_angle_sc2], [], 0],
                [[10.5, 10, 1, 'cC'], [start_angle_sc1], [], 0],
                [[10.5, 10, 2, 'cC'], [start_angle_sc2], [], 0],
                [[14.5, 10, 1, 'cF1'], [start_angle_sc1], [], 0],
                [[14.5, 10, 2, 'cF1'], [start_angle_sc2], [], 0],
                [[18.5, 10, 1, 'c2'], [start_angle_sc1], [], 0],
                [[18.5, 10, 2, 'c2'], [start_angle_sc2], [], 0],
                [[2.5, 14, 1, 'uN2'], [start_angle_sc1], [], 0],
                [[2.5, 14, 2, 'uN2'], [start_angle_sc2], [], 0],
                [[6.5, 14, 1, 'uN1'], [start_angle_sc1], [], 0],
                [[6.5, 14, 2, 'uN1'], [start_angle_sc2], [], 0],
                [[10.5, 14, 1, 'uC'], [start_angle_sc1], [], 0],
                [[10.5, 14, 2, 'uC'], [start_angle_sc2], [], 0],
                [[14.5, 14, 1, 'uF1'], [start_angle_sc1], [], 0],
                [[14.5, 14, 2, 'uF1'], [start_angle_sc2], [], 0],
                [[18.5, 14, 1, 'uF2'], [start_angle_sc1], [], 0],
                [[18.5, 14, 2, 'uF2'], [start_angle_sc2], [], 0]
            ]
        random.shuffle(sc_states)
        index_list = 25 * [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                        22, 23, 24, 25, 26, 27, 28, 29]
        random.shuffle(index_list)
        for i in range(0, 750):
            ind = index_list[i]
            this_sc = sc_states[ind][0]
            x = this_sc[0]
            x_sti.append(x)
            y = this_sc[1]
            y_sti.append(y)
            raw_loc = this_sc[3]
            raw_test_locs.append(raw_loc)
            ori_list = sc_states[ind][1]
            ori = ori_list[-1]
            orientations.append(ori)
            inputs = torch.tensor([[x, y, ori]])
            rho = torch.round(model_response(batch_size, p_no, inputs, condition)[0])
            perceived_tilt.append(rho.item())
            response_list = sc_states[ind][2]
            if ori > 90 and rho == 1:
                acc = 1
            elif ori < 90 and rho == 0:
                acc = 1
            else:
                acc = 0    
            response_list.append(acc)
            num_occurance = response_list.count(acc)
            stepsize_list = [2, 1, 0.5]
            if (len(response_list) > 1) and (sc_states[ind][3] < 2):
                if num_occurance == 1:
                    sc_states[ind][3] += 1
                    response_list.clear()
            stepsize = stepsize_list[sc_states[ind][3]]
            if acc == 1:
                if ori < 90:
                    next_ori = ori + stepsize
                else:
                    next_ori = ori - stepsize
            else:
                if ori < 90:
                    next_ori = ori - stepsize
                else:
                    next_ori = ori + stepsize
            sc_states[ind][1].append(next_ori)
            
        df = pd.DataFrame({
            'x_sti':x_sti,
            'y_sti':y_sti,
            'raw_test_loc': raw_test_locs,
            'theta': orientations,
            'rho': perceived_tilt
            })
        return df
        def forward(batch_size, p_no, condition):
            temp_df = experiment(batch_size, p_no, condition)
            df_exp = pd.DataFrame((temp_df), 
                                columns=['x_sti', 'y_sti', 'raw_test_loc', 'theta', 'rho'])
            df_exp.to_csv('./simulation_data/' + str(p_no) + '_' + '_' + condition + '_exp.csv', index=False)