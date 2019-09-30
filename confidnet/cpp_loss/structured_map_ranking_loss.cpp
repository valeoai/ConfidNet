/*
MIT License

Copyright (c) 2016 Thibaud Durand, Nicolas Thome and Matthieu Cord

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

https://github.com/durandtibo/mantra-python
*/

#include <torch/extension.h>

#include <iostream>
#include <vector>

at::Tensor generate_ranking_from_labels(at::Tensor &labels) {

    // initialize ranking vector
    at::Tensor ranking = at::zeros_like(labels);

    // define accessors
    auto labels_a = labels.accessor<long, 1>();
    auto ranking_a = ranking.accessor<long, 1>();

    auto num_examples = labels.size(0);

    // compute a ranking with respect to the given labels. It is possible to use another ranking.
    for (auto i = 0; i < num_examples; i++) {
        for (auto j = i + 1; j < num_examples; j++) {
            if (labels_a[i] == 1) {
                ranking_a[i] += 1;
                ranking_a[j] -= 1;
            } else {
                ranking_a[i] -= 1;
                ranking_a[j] += 1;
            }
        }
    }

    return ranking;
}

at::Tensor encode_ranking_cpu(at::Tensor &labels,
                              at::Tensor &positive_examples,
                              at::Tensor &negative_examples,
                              at::Tensor &example_index_map,
                              at::Tensor &optimum_loc_neg_example,
                              const int num_positive,
                              const int num_negative) {
    auto num_examples = labels.size(0);

    // initialize ranking vector
    at::Tensor ranking = at::zeros_like(labels);

    // define accessor
    auto labels_a = labels.accessor<long, 1>();
    auto ranking_a = ranking.accessor<long, 1>();
    auto positive_examples_a = positive_examples.accessor<float, 1>();
    auto negative_examples_a = negative_examples.accessor<float, 1>();
    auto example_index_map_a = example_index_map.accessor<long, 1>();
    auto optimum_loc_neg_example_a = optimum_loc_neg_example.accessor<long, 1>();

    // compute ranking
    for (auto i = 0; i < num_examples; i++) {
        auto label_i = labels_a[i];
        for (auto j = i + 1; j < num_examples; j++) {
            if (i == j) {
                // Nothing to do
            } else if (label_i == labels_a[j]) { // same label
                if (label_i == 1) { // positive labels
                    auto pos_ex_i = positive_examples_a[example_index_map_a[i]];
                    auto pos_ex_j = positive_examples_a[example_index_map_a[j]];
                    if (pos_ex_i > pos_ex_j) {
                        ranking_a[i] += 1;
                        ranking_a[j] -= 1;
                    } else if (pos_ex_j > pos_ex_i) {
                        ranking_a[i] -= 1;
                        ranking_a[j] += 1;
                    } else if (i < j) {
                        ranking_a[i] += 1;
                        ranking_a[j] -= 1;
                    } else {
                        ranking_a[i] -= 1;
                        ranking_a[j] += 1;
                    }
                } else { // negative labels
                    auto neg_ex_i = negative_examples_a[example_index_map_a[i]];
                    auto neg_ex_j = negative_examples_a[example_index_map_a[j]];
                    if (neg_ex_i > neg_ex_j) {
                        ranking_a[i] += 1;
                        ranking_a[j] -= 1;
                    } else if (neg_ex_j > neg_ex_i) {
                        ranking_a[i] -= 1;
                        ranking_a[j] += 1;
                    } else if (i < j) {
                        ranking_a[i] += 1;
                        ranking_a[j] -= 1;
                    } else {
                        ranking_a[i] -= 1;
                        ranking_a[j] += 1;
                    }
                }
            } else if (label_i == 1) {
                auto i_prime = example_index_map_a[i] + 1;
                auto j_prime = example_index_map_a[j] + 1;
                auto oj_prime = optimum_loc_neg_example_a[j_prime - 1];

                if ((oj_prime - i_prime - 0.5) > 0) {
                    ranking_a[i] += 1;
                    ranking_a[j] -= 1;
                } else {
                    ranking_a[i] -= 1;
                    ranking_a[j] += 1;
                }
            } else if (label_i == 0) {
                auto i_prime = example_index_map_a[i] + 1;
                auto j_prime = example_index_map_a[j] + 1;
                auto oi_prime = optimum_loc_neg_example_a[i_prime - 1];

                if ((j_prime - oi_prime + 0.5) > 0) {
                    ranking_a[i] += 1;
                    ranking_a[j] -= 1;
                } else {
                    ranking_a[i] -= 1;
                    ranking_a[j] += 1;
                }
            }
        }
    }

    return ranking;
}

at::Tensor find_optimum_neg_locations_cpu(at::Tensor &labels,
                                          at::Tensor &positive_examples,
                                          at::Tensor &negative_examples,
                                          at::Tensor &example_index_map,
                                          const int num_positive,
                                          const int num_negative) {

    auto max_value = 0.0;
    auto current_value = 0.0;
    auto max_index = -1;

    at::Tensor optimum_loc_neg_example = at::zeros_like(negative_examples).toType(at::kLong);

    // define accessors
    auto positive_examples_a = positive_examples.accessor<float, 1>();
    auto negative_examples_a = negative_examples.accessor<float, 1>();
    auto optimum_loc_neg_example_a = optimum_loc_neg_example.accessor<long, 1>();

    // find location of negative examples
    for (int j = 1; j <= num_negative; j++) {
        max_value = 0;
        max_index = num_positive + 1;
        // k is what we are maximising over. There would be one k_max for each negative example j
        current_value = 0;
        for (int k = num_positive; k >= 1; k--) {
            auto diff = positive_examples_a[k - 1] - negative_examples_a[j - 1];
            current_value +=
                    (1. / (float) num_positive) * ((j / (float) (j + k)) - ((j - 1) / (float) (j + k - 1))) -
                    (2. / (float) (num_positive * num_negative)) * diff;
            if (current_value > max_value) {
                max_value = current_value;
                max_index = k;
            }
        }
        optimum_loc_neg_example_a[j - 1] = max_index;
    }

    // encode ranking
    at::Tensor ranking = encode_ranking_cpu(labels, positive_examples, negative_examples, example_index_map,
                                            optimum_loc_neg_example, num_positive, num_negative);

    return ranking;
}

at::Tensor loss_augmented_prediction_cpu(at::Tensor &x, at::Tensor &y) {
    auto batch_size = x.size(0);
    auto num_positive = at::Scalar(y.nonzero().size(0)).toInt();
    auto num_negative = batch_size - num_positive;

    std::tuple <at::Tensor, at::Tensor> result = x.sort(0, true);
    at::Tensor sorted_values = std::get<0>(result);
    at::Tensor sorted_indices = std::get<1>(result);

    auto positive_id = 0;
    auto negative_id = 0;
    at::Tensor example_index_map = at::zeros_like(x).toType(at::kLong);
    at::Tensor positive_examples = at::zeros_like(x).resize_(num_positive);
    at::Tensor negative_examples = at::zeros_like(x).resize_(num_negative);

    auto y_a = y.accessor<long, 1>(); // other solution: auto y_p = y.data<int>();
    auto example_index_map_a = example_index_map.accessor<long, 1>();
    auto sorted_indices_a = sorted_indices.accessor<long, 1>();
    auto sorted_values_a = sorted_values.accessor<float, 1>();

    auto positive_examples_a = positive_examples.accessor<float, 1>();
    auto negative_examples_a = negative_examples.accessor<float, 1>();

    for (auto i = 0; i < batch_size; i++) {
        auto j = sorted_indices_a[i];
        auto yj = y_a[j];
        if (yj == 1) {
            example_index_map_a[j] = positive_id;
            positive_examples_a[positive_id] = sorted_values_a[i];
            positive_id += 1;
        } else {
            example_index_map_a[j] = negative_id;
            negative_examples_a[negative_id] = sorted_values_a[i];
            negative_id += 1;
        }
    }

    at::Tensor ranking = find_optimum_neg_locations_cpu(y, positive_examples, negative_examples, example_index_map,
                                                        num_positive, num_negative);
    return ranking;
}

at::Scalar compute_ranking_score_cpu(at::Tensor &scores, at::Tensor &labels, at::Tensor &ranking) {
    // initialize ranking vector
    at::Tensor count = at::zeros_like(scores);

    // define accessor
    auto labels_a = labels.accessor<long, 1>();
    auto count_a = count.accessor<float, 1>();
    auto ranking_a = ranking.accessor<long, 1>();
    auto scores_a = scores.accessor<float, 1>();

    auto num_examples = labels.size(0);

    for (auto i = 0; i < num_examples; i++) {
        if (labels_a[i] == 1) {
            for (auto j = 0; j < num_examples; j++) {
                if (labels_a[j] == 0) {
                    if (ranking_a[i] > ranking_a[j]) {
                        count_a[i] += 1;
                        count_a[j] -= 1;
                    } else {
                        count_a[i] -= 1;
                        count_a[j] += 1;
                    }
                }
            }
        }
    }

    auto score = 0.;
    auto num_pos = 0.;
    auto num_neg = 0.;
    for (auto i = 0; i < num_examples; i++) {
        score += scores_a[i] * count_a[i];
        if (labels_a[i] == 1) {
            num_pos += 1;
        } else {
            num_neg += 1;
        }
    }
    return score / (num_pos * num_neg);
}

at::Scalar average_precision_cpu(at::Tensor &labels, at::Tensor &scores) {

    auto num_examples = labels.size(0);

    // Store rank of all examples
    auto ranking = at::zeros_like(labels);
    // Store the list of examples sorted by rank. Higher rank to lower rank
    auto sorted_examples = at::zeros_like(labels);

    // define accessor
    auto labels_a = labels.accessor<long, 1>();
    auto scores_a = scores.accessor<float, 1>();
    auto ranking_a = ranking.accessor<long, 1>();
    auto sorted_examples_a = sorted_examples.accessor<long, 1>();

    // Convert rank matrix to rank list
    for (auto i = 0; i < num_examples; i++) {
        ranking_a[i] = 1;
        for (auto j = 0; j < num_examples; j++) {
            if (scores_a[i] > scores_a[j]) {
                ranking_a[i] += 1;
            }
        }
        sorted_examples_a[num_examples - ranking_a[i]] = i;
    }

    // Compute prec@i
    auto pos_count = 0.;
    auto total_count = 0.;
    auto precision_at_i = 0.;

    for (auto i = 0; i < num_examples; i++) {
        auto label = labels_a[sorted_examples_a[i]];
        if (label == 1) {
            pos_count += 1;
        }
        total_count += 1;
        if (label == 1) {
            precision_at_i += pos_count / total_count;
        }
    }
    if (pos_count == 0) {
        return 0.0;
    }
    precision_at_i /= pos_count;
    return precision_at_i;
}

at::Scalar num_positive_cpu(at::Tensor &labels) {

    // define accessor
    auto labels_a = labels.accessor<long, 1>();

    auto num_examples = labels.size(0);
    auto num_pos = 0.;
    for (auto i = 0; i < num_examples; i++) {
        if (labels_a[i] == 1) {
            num_pos += 1;
        }
    }
    return num_pos;
}

std::tuple <at::Scalar, at::Tensor> structured_map_ranking_loss_per_class_forward_cpu(at::Tensor &x, at::Tensor &y) {

    auto num_examples = y.size(0);
    auto num_positive = num_positive_cpu(y).toInt();
    auto num_negative = num_examples - num_positive;

    if (num_positive == 0 || num_negative == 0) {
        at::Tensor ranking = at::zeros_like(y);
        return std::make_tuple(0., ranking);
    }

    // loss augmented prediction
    at::Tensor ranking_lai = loss_augmented_prediction_cpu(x, y);
    auto score_lai = compute_ranking_score_cpu(x, y, ranking_lai);

    // predict a ranking for the ground truth
    at::Tensor ranking_gt = generate_ranking_from_labels(y);
    auto score_gt = compute_ranking_score_cpu(x, y, ranking_gt);

    // compute the AP
    auto scores_lai = ranking_lai.toType(at::kFloat); // generate scores based on ranking to evaluate AP
    auto ap = average_precision_cpu(y, scores_lai);

    // loss value
    auto loss_value = 1.0 - ap.toFloat() + score_lai.toFloat() - score_gt.toFloat();
    //std::cout << "score_gt=" << score_gt << "\t score_lai=" << score_lai << "\t ap=" << ap << std::endl;

    return std::make_tuple(loss_value, ranking_lai);
}

/***
 *
 * @param input
 * @param target +1 label positif / 0 label unknown / -1 label negatif
 * @return
 */
std::vector <at::Tensor> structured_map_ranking_loss_forward(at::Tensor input, at::Tensor target, at::Tensor mask) {

    auto num_categories = input.size(1);

    auto is_cuda = input.type().is_cuda();
    if (is_cuda) {
        input = input.toBackend(at::Backend::CPU);
        target = target.toBackend(at::Backend::CPU);
        mask = mask.toBackend(at::Backend::CPU);
    }

    // initialize outputs
    auto loss = at::zeros_like(input).resize_(num_categories);
    auto ranking_lai = at::zeros_like(target);

    // Compute the structured AP loss for each category
    for (auto i = 0; i < num_categories; i++) {
        auto x = input.select(1, i).squeeze();
        auto y = target.narrow(1, i, 1).squeeze().toType(at::kLong);
        auto m = mask.select(1, i);

        x = x.masked_select(m);
        y = y.masked_select(m);

        auto outputs = structured_map_ranking_loss_per_class_forward_cpu(x, y);

        loss[i] = std::get<0>(outputs);

        at::Tensor r = std::get<1>(outputs).toType(at::kFloat);
        ranking_lai.select(1, i).masked_scatter_(m, r);
    }
    loss = loss.sum() / (float) num_categories;

    if (is_cuda) {
        loss = loss.toBackend(at::Backend::CUDA);
        ranking_lai = ranking_lai.toBackend(at::Backend::CUDA);
    }

    return {loss, ranking_lai};
}

at::Tensor compute_ranking_gradient_cpu(at::Tensor &labels, at::Tensor &ranking) {
    // initialize gradient
    at::Tensor grad = at::zeros_like(labels).toType(at::kFloat);

    // define accessor
    auto labels_a = labels.accessor<long, 1>();
    auto grad_a = grad.accessor<float, 1>();
    auto ranking_a = ranking.accessor<long, 1>();

    // compute gradient
    auto num_examples = labels.size(0);
    for (auto i = 0; i < num_examples; i++) {
        if (labels_a[i] == 1) {
            for (auto j = 0; j < num_examples; j++) {
                if (labels_a[j] == 0) {
                    if (ranking_a[i] > ranking_a[j]) {
                        grad_a[i] += 1;
                        grad_a[j] -= 1;
                    } else {
                        grad_a[i] -= 1;
                        grad_a[j] += 1;
                    }
                }
            }
        }
    }

    return grad;
}

at::Tensor structured_map_ranking_loss_per_class_cpu_backward(at::Tensor &input,
                                                              at::Tensor &target,
                                                              at::Tensor &ranking_lai) {

    auto num_examples = target.size(0);
    auto num_positive = num_positive_cpu(target).toInt();
    auto num_negative = num_examples - num_positive;

    at::Tensor grad_input = at::zeros_like(input);

    if (num_positive == 0 || num_negative == 0) {
        return grad_input;
    }

    // compute gradient for the lai prediction
    grad_input += compute_ranking_gradient_cpu(target, ranking_lai);

    // predict a ranking for the ground truth
    at::Tensor ranking_gt = generate_ranking_from_labels(target);

    // compute gradient for the GT ranking
    grad_input -= compute_ranking_gradient_cpu(target, ranking_gt);

    grad_input /= (num_positive * num_negative);

    return grad_input;
}


at::Tensor structured_map_ranking_loss_backward(at::Tensor grad_output,
                                                at::Tensor input,
                                                at::Tensor target,
                                                at::Tensor mask,
                                                at::Tensor ranking_lai) {
    auto num_categories = input.size(1);

    auto is_cuda = input.type().is_cuda();
    if (is_cuda) {
        input = input.toBackend(at::Backend::CPU);
        target = target.toBackend(at::Backend::CPU);
        ranking_lai = ranking_lai.toBackend(at::Backend::CPU);
        mask = mask.toBackend(at::Backend::CPU);
    }

    at::Tensor grad_input = at::zeros_like(input);

    for (auto i = 0; i < num_categories; i++) {
        auto x = input.select(1, i).squeeze();
        auto y = target.narrow(1, i, 1).squeeze().toType(at::kLong);
        auto r = ranking_lai.narrow(1, i, 1).squeeze().toType(at::kLong);
        auto m = mask.select(1, i);

        x = x.masked_select(m);
        y = y.masked_select(m);
        r = r.masked_select(m);

        at::Tensor gi = structured_map_ranking_loss_per_class_cpu_backward(x, y, r);
        grad_input.select(1, i).masked_scatter_(m, gi);
    }

    if (is_cuda) {
        grad_input = grad_input.toBackend(at::Backend::CUDA);
    }

    grad_input *= grad_output / (float) num_categories;

    return grad_input;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("forward", &structured_map_ranking_loss_forward, "Structured Average Precision ranking loss forward");
m.def("backward", &structured_map_ranking_loss_backward, "Structured Average Precision ranking loss backward");
}