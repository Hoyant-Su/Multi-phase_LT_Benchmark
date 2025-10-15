# import pandas as pd

# def process_tumor_data_advanced(input_file, output_file):
#     """
#     根据新的复杂策略处理肿瘤数据。
#     """
#     try:
#         # 1. 加载数据
#         df = pd.read_csv(input_file)
#         print("--- 原始数据加载成功 ---")
#         print(df.head())
#     except FileNotFoundError:
#         print(f"错误: 输入文件 '{input_file}' 未找到。")
#         return

#     # 2. 预计算
#     # a. 创建基础ID列
#     df['Base_Tumor_ID'] = df['TumorID'].str.split('_tumor').str[0]
    
#     # b. 在任何筛选前，计算每个基础ID的平均Volume_mm3
#     df['Avg_Volume_mm3'] = df.groupby('Base_Tumor_ID')['Volume_mm3'].transform('mean')

#     # 3. 筛选主要数据 (arterial phase hyperenhancement 非空)
#     # 使用 pd.notna() 来判断非空
#     df_main_candidates = df[pd.notna(df['arterial phase hyperenhancement'])].copy()

#     # 为了找到每组最大的，先按Volume_mm3降序排序，然后去重保留第一个
#     df_main_candidates = df_main_candidates.sort_values('Volume_mm3', ascending=False)
#     df_main = df_main_candidates.drop_duplicates(subset=['Base_Tumor_ID'], keep='first')
    
#     print("\n--- 步骤1: 处理 'arterial phase hyperenhancement' 非空的行 ---")
#     print(df_main[['TumorID', 'Volume_mm3', 'arterial phase hyperenhancement', 'Avg_Volume_mm3']].to_string())

#     # 4. 处理补充数据
#     # a. 获取已处理过的基础ID集合
#     processed_ids = set(df_main['Base_Tumor_ID'])

#     # b. 从原始数据中筛选出那些基础ID未被处理的行
#     df_supplement_candidates = df[~df['Base_Tumor_ID'].isin(processed_ids)].copy()

#     # c. 对于这些补充行，找到每个基础ID中Volume_mm3最大的那一行
#     if not df_supplement_candidates.empty:
#         max_volume_indices = df_supplement_candidates.groupby('Base_Tumor_ID')['Volume_mm3'].idxmax()
#         df_supplement = df.loc[max_volume_indices]
#         print("\n--- 步骤2: 补充未被纳入的基础ID中体积最大的一行 ---")
#         print(df_supplement[['TumorID', 'Volume_mm3', 'Avg_Volume_mm3']].to_string())
#     else:
#         df_supplement = pd.DataFrame() # 如果没有补充数据，则创建一个空的DataFrame
#         print("\n--- 步骤2: 没有需要补充的数据 ---")


#     # 5. 合并结果
#     final_df = pd.concat([df_main, df_supplement], ignore_index=True)
    
#     # 6. 最后整理
#     # a. 调整列顺序: 将 Avg_Volume_mm3 插入到第三个位置
#     cols = final_df.columns.to_list()
#     avg_col = cols.pop(cols.index('Avg_Volume_mm3'))
#     cols.insert(2, avg_col)
#     final_df = final_df[cols]

#     # b. 移除不再需要的列
#     if 'InExp' in final_df.columns:
#         final_df = final_df.drop(columns=['InExp'])
#     final_df = final_df.drop(columns=['Base_Tumor_ID'])

#     # 7. 保存结果
#     final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
#     print(f"\n--- 处理完成！结果已保存到 '{output_file}' ---")
#     print("最终数据表:")
#     print(final_df.to_string())


# # --- 脚本执行入口 ---
# if __name__ == "__main__":
#     input_filename = '/inspire/hdd/project/continuinglearning/suhaoyang-240107100018/suhaoyang-240107100018/storage/Multi_phase_LT_benchmark/meta/draft_version/meta_tumor_level.csv'
#     output_filename = "/inspire/hdd/project/continuinglearning/suhaoyang-240107100018/suhaoyang-240107100018/storage/Multi_phase_LT_benchmark/meta/meta_tumor_level.csv"
    
#     process_tumor_data_advanced(input_filename, output_filename)

import os
import pandas as pd


csv_path = "/inspire/hdd/project/continuinglearning/suhaoyang-240107100018/suhaoyang-240107100018/storage/Multi_phase_LT_benchmark/meta/meta_info_tumor.csv"
import pandas as pd

# 读取原始 CSV
df = set(pd.read_csv(csv_path)['TumorID'])


csv_path2 = "/inspire/hdd/project/continuinglearning/suhaoyang-240107100018/suhaoyang-240107100018/storage/Multi_phase_LT_benchmark/meta/draft_version/notes.csv"
df2 = set(pd.read_csv(csv_path2)['ID'])
df_now = set.union(df, df2)

print(set(os.listdir("/inspire/hdd/project/continuinglearning/suhaoyang-240107100018/suhaoyang-240107100018/storage/tumor_radiomics/Data/final_volume")) - df)

