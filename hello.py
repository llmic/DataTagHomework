import os
import regex as re
import pandas as pd
import ollama
from typing import List, Tuple, Optional, Dict

#应特定规则而生成的标注规则，被标注文件可改）
# ====================== 核心配置区（100%对齐作业要求，禁止随意修改）======================
# 1. 官方核心关键词词典（与作业要求完全一致，长词优先匹配，无遗漏/新增）
KEYWORDS = [
    # 基础与连接
    "设备上云", "物联网", "设备联网", "自动信息采集", "数字化设备", "数字化改造",
    "工业协议解析", "5G", "标识解析", "数据共享", "内外网络改造",
    # 平台与数据
    "工业互联网", "云平台", "云ERP", "云端部署", "大数据平台", "数据中台",
    "数据标准", "数据质量", "元数据", "主数据", "数据资产", "数据挖掘",
    "数据可视化", "数据融合", "数据开放共享", "分类分级存储", "工业大数据",
    # 智能应用与技术
    "工业APP", "云计算", "边缘计算", "边缘节点", "边云协同", "数字孪生",
    "机器学习", "人工智能", "智能计算", "自适应控制", "实时控制", "实时采集",
    "低时延传输", "本地存储", "知识图谱", "工业知识", "知识沉淀", "知识复用",
    "工业机理模型", "模型化", "软件化", "微服务", "微组件", "算法模型", "智能算法",
    # 生产与制造
    "智能制造", "智能化生产", "智能工厂", "柔性生产", "敏捷制造", "实时调度",
    "智能排产", "工艺优化", "在线检测", "智能检测", "质量管控", "生产过程优化",
    "设备状态监测", "预测性维护", "设备故障诊断", "能耗优化", "远程运维", "设备健康管理",
    # 管理与模式创新
    "数字化管理", "数字化转型", "数据驱动", "流程再造", "在线办公", "线上培训",
    "智能决策", "网络化协同", "协同设计", "协同制造", "协同研发", "产业链协同",
    "供应链协同", "信息共享", "业务协作", "柔性配置", "个性化定制", "客户画像",
    "模块化设计", "敏捷研发", "全流程参与", "服务化延伸", "产品增值", "制造能力共享",
    "在线交易", "产融合作", "融资租赁", "创业创新",
    # 安全
    "信息安全", "安全防护", "风险评估", "监测预警", "应急响应", "数据安全", "数据防篡改",
    # 效益
    "研发效率", "生产效率", "产能利用率", "运维成本", "运营成本", "产品质量",
    "良品率", "库存周转", "交货期", "客户满意度", "劳动生产率"
]
# 去重+长词优先匹配（避免短词覆盖长词，匹配更精准）
KEYWORDS = list(set(KEYWORDS))
KEYWORDS.sort(key=lambda x: len(x), reverse=True)

# 2. 官方标注规则系统提示词（严格对齐作业要求，新增上下文适配+原因输出）
LABEL_SYSTEM_PROMPT = """
你是一个严格遵守作业规则的工业互联网应用文本标注助手，所有判断必须100%基于给定的原文内容和以下官方规则，禁止主观推断。

【核心标注规则（完全来自作业要求，无任何自定义修改）】
=== 标注为1的强制条件（必须同时全部满足，缺一不可）：
1. 待标注句子必须包含指定的工业互联网核心关键词（必填前提）；
2. 自我应用属性：描述的是**本企业自身**在生产、研发、管理、供应链、运维等内部核心业务流程的改造优化，而非对外销售产品/服务、对外技术赋能、建设平台供第三方使用、参与平台标准制定等乙方行为；
3. 已落地实施：包含明确的**已完成、已落地**的应用行为动词，如「部署了、采用了、实施了、应用于、搭建了、实现了、通过...优化了、通过...提升了」等；
4. 无未来/未完成表述：句子中绝对不包含「拟、计划、将、有望、探索、布局、推进、加快、加强」等表未来规划或未落地的词汇；
5. 主语明确：句子主语明确指向本企业，而非「行业、业内、相关企业」等模糊主语；
6. 最终用户属性：企业是作为工业互联网平台的**最终使用方**，在核心业务环节调用平台功能，而非仅自建平台、开发平台、提供平台技术服务。

=== 标注为0的条件（满足任意一条即直接判定为0）：
1. 待标注句子不包含指定的工业互联网核心关键词；
2. 属于对外服务/销售、对外技术赋能、建设平台供第三方使用、参与标准制定等乙方行为，而非企业自身内部应用；
3. 仅描述专利、技术研发、理论成果，未明确说明已实际落地应用到企业自身核心业务；
4. 属于战略规划、未来计划，包含「拟、计划、将、有望、探索、布局、推进、加快、加强」等未来时态/未完成状态的词汇；
5. 主语模糊，未明确指向本企业的实际落地行为；
6. 仅描述通用数字化系统（如普通ERP、CRM等），无工业互联网相关的实际落地应用；
7. 属于非工业领域的应用，与工业互联网场景无关；
8. 无明确的已落地应用行为动词，无实际成效的模糊性描述；
9. 企业仅为平台建设方、服务提供方，而非平台最终使用方。

【标注要求】
1. 必须严格结合给定的【句子上下文】和【待标注句子】进行判断，不得脱离原文主观推断，所有判断必须有原文依据；
2. 先输出标注推理原因，再输出最终标注结果，格式必须严格遵守以下要求，不得有任何额外内容；
3. 推理原因必须明确说明符合哪条标注规则，有原文对应依据，不得模糊表述。

【严格输出格式】
标注原因：[你的推理过程和规则依据，必须清晰明确]
标注结果：X
（其中X仅能为0或1，不得有其他字符、空格、标点）
"""

# 3. 作业要求配置（禁止修改，完全匹配作业要求）
OUTPUT_FOLDER = "./单年报标注结果"  # 单份年报标注结果存放路径
SUMMARY_OUTPUT_PATH = "./班级+学号+姓名_总标注结果.xlsx"  # 最终提交的汇总文件
TEMP_SAVE_PATH = "./临时标注进度.csv"  # 临时保存路径，防止程序中断丢失数据
CONTEXT_WINDOW = 5  # 上下文窗口：待标注句前后各5句
# ==================================================================================


def split_sentences(text: str) -> List[str]:
    """优化版中文分句函数"""
    text = re.sub(r'[\n\t\s\u3000]+', ' ', text.strip())
    sentence_end_pattern = r'(?<![（(][^）)]*)([。！？；])'
    sentences = re.split(sentence_end_pattern, text)
    full_sentences = []
    for i in range(0, len(sentences)-1, 2):
        full_sent = sentences[i].strip() + sentences[i+1].strip()
        if len(full_sent) >= 10:
            full_sentences.append(full_sent)
    if len(sentences) % 2 == 1 and len(sentences[-1].strip()) >= 10:
        full_sentences.append(sentences[-1].strip())
    return full_sentences


def load_all_candidates(folder_path: str) -> List[Dict]:
    """加载所有年报的所有候选句，生成全局待标注队列"""
    all_candidates = []
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    print(f"\n📂 正在加载所有年报文件，共发现 {len(txt_files)} 份...")
    
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        
        # 解析文件名
        stock_id, year, company_name = None, None, None
        standard_pattern = r'^(\d{6})_(\d{4})_([^_]+)_.*\.txt$'
        match = re.match(standard_pattern, txt_file)
        if match:
            stock_id, year, company_name = match.group(1), match.group(2), match.group(3)
        else:
            simple_pattern = r'^(\d{6})_(\d{4})_.*\.txt$'
            simple_match = re.match(simple_pattern, txt_file)
            if simple_match:
                stock_id, year, company_name = simple_match.group(1), simple_match.group(2), "未知企业"
            else:
                stock_id, year, company_name = "000000", "0000", "未知企业"

        # 读取文件
        text = None
        encoding_list = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'cp1252']
        for encoding in encoding_list:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if text is None:
            print(f"⚠️  无法读取文件，跳过：{txt_file}")
            continue

        # 分句+筛选候选句
        all_sentences = split_sentences(text)
        total_sentences = len(all_sentences)
        
        for idx, sent in enumerate(all_sentences):
            has_keyword = any(kw in sent for kw in KEYWORDS)
            if has_keyword:
                # 提取上下文
                start_idx = max(0, idx - CONTEXT_WINDOW)
                end_idx = min(total_sentences, idx + CONTEXT_WINDOW + 1)
                context = "\n".join(all_sentences[start_idx:end_idx])
                
                # 加入全局队列
                all_candidates.append({
                    "sentence": sent,
                    "context": context,
                    "source_file": txt_file,
                    "stock_id": stock_id,
                    "year": year,
                    "company_name": company_name,
                    "is_labeled": False,
                    "label": -1,
                    "reason": ""
                })
    
    print(f"✅ 加载完成！共生成 {len(all_candidates)} 条待标注句子\n")
    return all_candidates


def label_single_sentence(
    sentence: str, 
    context: str,
    model_name: str = "deepseek-r1:8b", 
    max_retry: int = 3
) -> Tuple[int, str]:
    """单句标注函数，带重试机制"""
    user_prompt = f"""
【句子上下文】：
{context}
【待标注句子】：
{sentence}
"""
    
    for retry in range(max_retry):
        full_response = ""
        try:
            stream = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": LABEL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
                options={"temperature": 0, "top_p": 0.01, "num_predict": 1024}
            )

            for chunk in stream:
                content = chunk['message']['content']
                print(content, end='', flush=True)
                full_response += content

            # 提取结果
            reason_match = re.search(r'标注原因[：:]\s*(.*?)(?=\n标注结果)', full_response, re.DOTALL)
            label_reason = reason_match.group(1).strip() if reason_match else "未提取到有效推理原因"
            
            result_match = re.search(r'标注结果[：:]\s*([01])', full_response)
            if result_match:
                return int(result_match.group(1)), label_reason
            else:
                digits = re.findall(r'[01]', full_response)
                if digits:
                    return int(digits[-1]), label_reason

        except Exception as e:
            print(f"\n❌ 出错，重试 {retry+1}/{max_retry} | 错误：{str(e)}")

    return -1, "模型调用多次失败，需人工手动标注"


def save_temp_progress(candidates: List[Dict]):
    """保存临时进度到CSV，防止中断丢失"""
    df = pd.DataFrame(candidates)
    df.to_csv(TEMP_SAVE_PATH, index=False, encoding='utf-8-sig')


def load_temp_progress() -> Optional[List[Dict]]:
    """尝试加载临时进度"""
    if os.path.exists(TEMP_SAVE_PATH):
        df = pd.read_csv(TEMP_SAVE_PATH)
        return df.to_dict('records')
    return None


def main():
    # ====================== 运行配置（仅需修改此处）======================
    REPORT_FOLDER = "./data_source"
    MODEL_NAME = "deepseek-r1:8b"
    # ======================================================================

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. 加载待标注队列（优先加载临时进度）
    candidates = load_temp_progress()
    if candidates is None:
        candidates = load_all_candidates(REPORT_FOLDER)
    else:
        unlabeled_count = sum(1 for c in candidates if not c['is_labeled'])
        print(f"\n🔄 检测到临时进度，已加载 {len(candidates)} 条句子，其中 {unlabeled_count} 条未标注\n")

    # 2. 逐条自动化标注
    total_count = len(candidates)
    for i, item in enumerate(candidates):
        if item['is_labeled']:
            continue  # 跳过已标注的句子

        # 显示进度和待标注句子
        print(f"\n{'#'*80}")
        print(f"🚀 标注进度：{i+1}/{total_count}")
        print(f"📂 来源文件：{item['source_file']}")
        print(f"📝 待标注句子：{item['sentence']}")
        print(f"🤖 模型推理过程：")
        print("-"*80)

        # 执行标注
        label_result, label_reason = label_single_sentence(
            sentence=item['sentence'],
            context=item['context'],
            model_name=MODEL_NAME
        )

        # 更新结果
        item['label'] = label_result
        item['reason'] = label_reason
        item['is_labeled'] = True

        # 显示结果
        print("-"*80)
        print(f"✅ 标注完成 | 结果：{label_result}")
        print(f"📝 标注原因：{label_reason[:100]}{'...' if len(label_reason) > 100 else ''}")
        print(f"💾 正在保存临时进度...")

        # 每标注完一句就保存一次进度，绝对安全
        save_temp_progress(candidates)
        print(f"✅ 临时进度已保存\n")

    # 3. 全部标注完成，生成最终Excel
    print(f"\n{'#'*80}")
    print(f"🎉 所有句子标注完成！正在生成最终文件...")
    print(f"{'#'*80}")

    # 整理结果
    final_result = []
    for item in candidates:
        final_result.append({
            "句子内容": item['sentence'],
            "来源文件": item['source_file'],
            "人工标注标签": item['label'],
            "id": item['stock_id'],
            "year": item['year'],
            "原因": item['reason']
        })

    final_df = pd.DataFrame(final_result)

    # 生成汇总文件
    final_df.to_excel(SUMMARY_OUTPUT_PATH, index=False)
    print(f"\n✅ 最终提交汇总文件已生成：{os.path.abspath(SUMMARY_OUTPUT_PATH)}")

    # 按文件拆分保存
    for source_file in final_df['来源文件'].unique():
        file_df = final_df[final_df['来源文件'] == source_file]
        # 从文件名提取信息用于命名
        stock_id = file_df['id'].iloc[0]
        year = file_df['year'].iloc[0]
        # 简单提取企业名
        company_name = "未知企业"
        match = re.match(r'^\d{6}_\d{4}_([^_]+)_.*\.txt$', source_file)
        if match:
            company_name = match.group(1)
        
        excel_filename = f"{stock_id}_{year}_{company_name}_标注结果.xlsx"
        excel_path = os.path.join(OUTPUT_FOLDER, excel_filename)
        file_df.to_excel(excel_path, index=False)
        print(f"✅ 单文件已生成：{os.path.abspath(excel_path)}")

    # 删除临时文件
    if os.path.exists(TEMP_SAVE_PATH):
        os.remove(TEMP_SAVE_PATH)
        print(f"\n✅ 临时进度文件已清理")

    print(f"\n{'#'*80}")
    print(f"🏆 全部工作完成！")
    print(f"{'#'*80}")


if __name__ == "__main__":
    main()