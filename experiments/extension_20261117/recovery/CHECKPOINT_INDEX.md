# Checkpoint index

Four verified additions; original twelve results, resource records and checkpoint coverage retained unchanged. Local metadata conversion only; no server access or checkpoint loading.

Recorded checkpoints: 16/16. New verified additions: 4/4.

| Dataset | Method / variant | Storage kind | Host alias | MiB | SHA256 | Native CPU restore |
| --- | --- | --- | --- | ---: | --- | --- |
| cifar224 | shared_basis / original | existing_private_workspace_archive | legacy index | 335.40 | `1b298fbb052c01fef00c2797fe803e1f7164dfc33decd09a132483cac7b4df88` | false |
| cifar224 | history_transport / original | verified_server_local_backup | legacy index | 686.45 | `333345b4de76e3e213401ee5c1452a91ab5fba4693e05dd1eca02d14fb375c4a` | true |
| cifar224 | current_adapted / epochs5 | verified_server_local_backup | legacy index | 1337.90 | `b842888aa0cf0a19d9b2a08b2d2f9ac00926f44fb1e5e92e458988db9da040e0` | true |
| cifar224 | plastic_shared_basis / original | verified_server_local_backup | legacy index | 694.25 | `1e11132f1705e32e27fefdd4281bfcad5a348a0eb916ece66595b3579081abe2` | true |
| cifar224 | er / original | verified_server_local_backup | legacy index | 336.40 | `71bd38161d5b4db6ff0767e38d2548f86c67bf5b77e1456836c3131e08e3fee8` | true |
| cifar224 | joint / original | existing_private_workspace_archive | legacy index | 327.65 | `4140bc5c9470bb29f90e58f08890ed3ca8829496da23dd08ff488a2007dec210` | false |
| imagenetr | shared_basis / original | existing_private_workspace_archive | legacy index | 336.37 | `ec502dc530ad179f4207ecbef688093d868d534a03122c00756e25c3fca7fd12` | false |
| imagenetr | history_transport / original | verified_server_local_backup | legacy index | 681.33 | `c24b44f387054bb8fcdfeb1d9a0f44dc30633fbffb5d7691c68ec82602c50d3a` | true |
| imagenetr | current_adapted / epochs5 | verified_server_local_backup | legacy index | 1343.66 | `a745a17777a7faf149bb8c4d1af7ca6c42caf1ac1ce318ba05d8ca3d8cc313e7` | true |
| imagenetr | plastic_shared_basis / original | verified_server_local_backup | legacy index | 690.26 | `2d20633df13a58716805510adce4c423ee9714a95ae40d25d9ba7c7c444097b2` | true |
| imagenetr | er / original | verified_server_local_backup | legacy index | 328.40 | `238b00f19af6198d4e1f81d44400e0e1e0373cc887f531d4e97e396ec298267d` | true |
| imagenetr | joint / original | existing_private_workspace_archive | legacy index | 327.94 | `531e84fb056b1f02ec999b87a1d907ff8e27a0931474545763f57dbc2849e57a` | false |
| cifar224 | current_adapted / epochs20 | verified_server_local_backup | current20_server | 1337.96 | `7ea3ea215ad9a0c3cb5235fb92a09bcae7301cb50faf519e47bf56b18187cf8f` | true |
| imagenetr | current_adapted / epochs20 | verified_server_local_backup | current20_server | 1343.69 | `3a2f9a19221606f697911800757a560d426da5e1ab77b851f84a189521025f82` | true |
| cifar224 | tuna / epochs20 | verified_server_local_backup | tuna_server | 568.09 | `4d7f1a3b109712e3f88d2deaf054ea52cba9c153aa25a76b25941468ffee28c3` | true |
| imagenetr | tuna / epochs20 | verified_server_local_backup | tuna_server | 793.80 | `cc7a8e84d013dad3aa90804cb74f9491ac9675c771d7a20e477a309d747ac8f5` | true |

The twelve original entries are copied unchanged, including their original audit coverage. Four use retained private workspace archives; the remaining original entries and the new additions use server-local backups. CPU restoration is not an additional accuracy run or a proof of identical future GPU trajectories.

## Locations

- `formal_cifar_shared_selected_v7`: `${PRIVATE_WORKSPACE_ROOT}/formal_checkpoints/formal_cifar_shared_selected_v7/checkpoint.pt`
- `formal_cifar_transport_selected_v2`: `${RUNTIME_ROOT}/formal_backups/campaign_20261117/formal_cifar_transport_selected_v2/checkpoint.pt`
- `formal_cifar_current_selected_v5`: `${RUNTIME_ROOT}/formal_backups/campaign_20261117/formal_cifar_current_selected_v5/checkpoint.pt`
- `formal_cifar_plastic_shared_selected_v5`: `${RUNTIME_ROOT}/formal_backups/campaign_20261117/formal_cifar_plastic_shared_selected_v5/checkpoint.pt`
- `formal_cifar_er_fixed_v1`: `${ROOT_BACKUP_ROOT}/campaign_20261117/formal_cifar_er_fixed_v1/checkpoint.pt`
- `formal_cifar_joint_fixed_v1`: `${PRIVATE_WORKSPACE_ROOT}/formal_checkpoints/formal_cifar_joint_fixed_v1/checkpoint.pt`
- `formal_imagenetr_shared_selected_v7`: `${PRIVATE_WORKSPACE_ROOT}/formal_checkpoints/formal_imagenetr_shared_selected_v7/checkpoint.pt`
- `formal_imagenetr_transport_selected_v3`: `${RUNTIME_ROOT}/formal_backups/campaign_20261117/formal_imagenetr_transport_selected_v3/checkpoint.pt`
- `formal_imagenetr_current_selected_v5`: `${RUNTIME_ROOT}/formal_backups/campaign_20261117/formal_imagenetr_current_selected_v5/checkpoint.pt`
- `formal_imagenetr_plastic_shared_selected_v5`: `${RUNTIME_ROOT}/formal_backups/campaign_20261117/formal_imagenetr_plastic_shared_selected_v5/checkpoint.pt`
- `formal_imagenetr_er_fixed_v1`: `${ROOT_BACKUP_ROOT}/campaign_20261117/formal_imagenetr_er_fixed_v1/checkpoint.pt`
- `formal_imagenetr_joint_fixed_v1`: `${PRIVATE_WORKSPACE_ROOT}/formal_checkpoints/formal_imagenetr_joint_fixed_v1/checkpoint.pt`
- `formal_cifar_current_epochs20_v1`: `/root/autodl-tmp/cl27/formal_backups/recovery_20261118_v2/formal_cifar_current_epochs20_v1/checkpoint.pt`
- `formal_imagenetr_current_epochs20_v1`: `/root/autodl-tmp/cl27/formal_backups/recovery_20261118_v2/formal_imagenetr_current_epochs20_v1/checkpoint.pt`
- `formal_cifar_tuna_native_v1`: `/root/autodl-tmp/cl27/formal_backups/recovery_20261118/formal_cifar_tuna_native_v1/checkpoint.pt`
- `formal_imagenetr_tuna_native_v1`: `/root/autodl-tmp/cl27/formal_backups/recovery_20261118/formal_imagenetr_tuna_native_v1/checkpoint.pt`

Legacy path-root mappings and environment/dependencies remain in the original index metadata embedded in the JSON. New paths are scoped by host_alias; identical server-local path strings on different hosts do not identify the same file. No checkpoint bytes were reopened, copied, or downloaded by this report generator.
