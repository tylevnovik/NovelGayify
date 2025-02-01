import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import ebooklib
import openai
import pandas as pd
from bs4 import BeautifulSoup
from ebooklib import epub
from tenacity import retry, stop_after_attempt, wait_exponential


class NovelGayifier:
    def __init__(self, epub_path: str, save_dir: str, api_key: str):
        self.epub_path = epub_path
        self.save_dir = save_dir
        self.api_key = api_key
        self.chapters_df = pd.DataFrame(columns=['chapter_id', 'title', 'content', 'processed_content'])
        self.character_mappings_df = pd.DataFrame(columns=['character_name', 'mapped_name', 'seen_chapters'])
        self.checkpoint_path = os.path.join(save_dir, f'{Path(epub_path).stem}.checkpoint.pkl')
        self.system_prompt = """你是一个小说性别转换助手。你的任务是:
                                1. 识别文本中的具名女性角色
                                2. 为每个女性角色创建合适的男性化名字和特征
                                3. 将文本中的内容改写为全男性视角，感情关系改写为男同性恋关系
                                4. 保持故事情节和写作风格不变
                                5. 将家庭关系转换为双父亲家庭模式
                                6. 以markdown格式输出转换后的内容，修改的地方用粗体标识出来

                                转换规则:
                                1. 涉及核心家庭关系代词的描述，母亲改为"爹爹"，父亲保持"爸爸"，其余家庭关系进行类似改写
                                2. 女性特有的形容词和行为要改写为男性化表达
                                    (1)生理特征转换规则：
                                        a. 体型描述：
                                           "纤细"→"精瘦" / "婀娜"→"挺拔"
                                        b. 动态特征：
                                           "轻咬下唇"→"紧抿嘴角"
                                           "绞着衣角"→"握紧剑柄"
                                        c. 保留10%中性特征（如"修长的手指"）
                                        d.转换示例：
                                            原句：她拢了拢如瀑青丝，眼波流转间尽是风情
                                            转换：他捋了捋束起的长发，眉峰微挑自带三分锐气
                                    (2)语言习惯转换规则：
                                        a. 找出含以下特征的对话：
                                           - 高频语气词（呢/呀/嘛）
                                           - 疑问句式占比＞40%
                                           - 包含感性比喻（如"心像被针扎"）
                                        b. 将每句话按以下规则重构：
                                           - 删除冗余修饰词（"真的非常"→"确实"）
                                           - 疑问句转陈述句（"不觉得痛吗？"→"忍着点"）
                                           - 增加行动指令（补充"走"、"动手"等动词）
                                        c.转换示例：
                                            原句："人家才不要穿这么丑的铠甲呢！"
                                            转换："这铠甲太碍事，换了。"
                                    (3)情感关系转换规则：
                                        a. 关系动力学调整：
                                           - 将"保护欲"转化为"竞争性默契"，也可以保留一部分保护欲
                                           - "情感依赖"改为"能力互补"，也可以保留一部分情感依赖
                                        b. 肢体接触转化表：
                                           "轻抚脸颊"→"拍肩停留"，也可以保留一部分摸脸
                                           "依偎怀中"→"背靠背喘息"，也可以保留一部分拥抱
                                        c. 保留关键情感节点，调整触发机制：
                                           原："他心疼她的眼泪" 
                                           改："他读懂了他眼底的执拗"
                                        d. 转换示例
                                            原句：她颤抖着解开衣带，月光洒在雪白的肩头...
                                            转换：他扯开染血的衣襟，月光勾勒出绷紧的肌肉线条...
                                3. 感情和亲密关系改写为男同关系
                                4. 保持人物性格特点，只改变性别相关的表达
                                5. 连贯性保障
                                    (1)上下文记忆："此前已将[旧角色名]改为[新角色名]，确保全文指代统一"
                                    (2)特征一致性检查："确认该角色在第三章描写的剑伤疤痕仍存在于第七章"
                                    (3)关系网验证："检查师徒关系中的年龄差是否符合转换后性别设定"

                                请特别注意：
                                1. 避免将男性特征等同于暴力倾向
                                2. 同性感情线需符合现代平等价值观
                                3. 保留角色脆弱性等人性化特质
                                4. 遇到敏感场景时主动标注建议人工复核
                                """

    def load_epub(self) -> None:
        """加载EPUB文件并解析为章节数据"""
        book = epub.read_epub(self.epub_path)
        chapters = []

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text_content = soup.get_text()
                if text_content.strip():  # 确保内容不为空
                    chapters.append({
                        'chapter_id': len(chapters),
                        'title': soup.title.string if soup.title else f"Chapter {len(chapters)}",
                        'content': text_content,
                    })

        # 假设 chapters 是一个包含新行的 DataFrame
        new_chapters_df = pd.DataFrame(chapters)

        # 使用 pd.concat 将新行添加到原始的 chapters_df 中
        self.chapters_df = pd.concat([self.chapters_df, new_chapters_df], ignore_index=True)

    def save_checkpoint(self) -> None:
        """保存处理进度"""
        checkpoint_data = {
            'chapters_df': self.chapters_df,
            'character_mappings_df': self.character_mappings_df
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

    def load_checkpoint(self) -> bool:
        """加载之前的处理进度"""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'rb') as f:
                data = pickle.load(f)
                self.chapters_df = data['chapters_df']
                self.character_mappings_df = data['character_mappings_df']
            return True
        return False

    def get_context_for_chapter(self, chapter_id: int) -> str:
        """获取当前章节的相关上下文"""
        current_chapter = self.chapters_df.iloc[chapter_id]
        context = []

        # 附上上一章的转换后内容
        if chapter_id > 0:
            prev_chapter = self.chapters_df.iloc[chapter_id - 1]
            if prev_chapter['processed_content'] is not None:
                context.append(prev_chapter['processed_content'])

        # 查找当前章节所有具名女角色之前最后一次出现的章节转换后的内容
        female_chars = self.extract_female_characters(current_chapter['content'])
        for char in female_chars:
            seen_chapters = self.character_mappings_df.loc[
                self.character_mappings_df['character_name'] == char, 'seen_chapters'
            ].values
            if seen_chapters.size > 0:
                last_seen_chapter_id = seen_chapters[0][-1]  # 取最后一个章节ID
                if last_seen_chapter_id < chapter_id:
                    last_seen_content = self.chapters_df.iloc[last_seen_chapter_id]['processed_content']
                    if last_seen_content:
                        context.append(last_seen_content)

        return "\n".join(context)

    def extract_female_characters(self, text: str) -> List[str]:
        """提取文本中的具名女性角色，并记录角色出现的章节ID"""
        prompt = f"""请分析以下文本，列出所有具名的女性角色名字（直接输出汉字，不要带任何其他符号）（每行一个）：

                    {text}

                    只需要返回名字列表，无需其他解释。"""

        response = self._call_openai(prompt)
        female_chars = [name.strip() for name in response.split('\n') if name.strip()]

        # 记录角色出现的章节ID
        for char in female_chars:
            if char in self.character_mappings_df['character_name'].values:
                self.character_mappings_df.loc[
                    self.character_mappings_df['character_name'] == char, 'seen_chapters'
                ].values[0].append(self.chapters_df[self.chapters_df['content'] == text].index[0])
            else:
                self.character_mappings_df = pd.concat([self.character_mappings_df, pd.DataFrame({
                    'character_name': char,
                    'mapped_name': '',
                    'seen_chapters': [[self.chapters_df[self.chapters_df['content'] == text].index[0]]]
                })], ignore_index=True)

        return female_chars

    def get_character_mapping(self, female_names: List[str]) -> Dict[str, str]:
        """为女性角色生成男性化的名字映射"""
        if not female_names:
            return {}

        names_str = '\n'.join(female_names)
        prompt = f"""请为以下女性角色名字设计对应的男性名字，姓氏不变，保持名字风格相似：
                    女→男用字对照：
                    柔→刚 / 婷→霆 / 芳→锋 
                    双名末字改为自然意象（雪→峰/雨→云）

                    {names_str}

                    请按以下格式返回映射（每行一个）：
                    原名=>新名"""

        response = self._call_openai(prompt)
        mappings = {}
        for line in response.split('\n'):
            if '=>' in line:
                old, new = line.split('=>')
                mappings[old.strip()] = new.strip()
                self.character_mappings_df.loc[
                    self.character_mappings_df['character_name'] == old.strip(), 'mapped_name'
                ] = new.strip()
        return mappings

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_openai(self, prompt: str, temperature: float = 0.7) -> str:
        """调用OpenAI API的封装方法"""
        client = openai.Client(api_key=self.api_key, base_url='https://api.deepseek.com/v1')
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    def process_chapter(self, chapter_id: int) -> None:
        """处理单个章节的性别转换"""
        current_chapter = self.chapters_df.iloc[chapter_id]
        context = self.get_context_for_chapter(chapter_id)

        # 1. 提取当前章节的女性角色
        logging.info(f'提取章节 {chapter_id} 的女性角色: {current_chapter["title"]}')
        female_chars = self.extract_female_characters(current_chapter['content'])

        # 2. 获取或更新角色映射
        logging.info(f'获取章节 {chapter_id} 的角色映射: {current_chapter["title"]}')
        new_mappings = self.get_character_mapping(female_chars)
        for old_name, new_name in new_mappings.items():
            self.character_mappings_df.loc[
                self.character_mappings_df['character_name'] == old_name, 'mapped_name'
            ] = new_name

        # 3. 构建转换提示
        mappings_str = ""
        for _, row in self.character_mappings_df.iterrows():
            mappings_str += f"{row['character_name']}=>{row['mapped_name']}\n"

        transform_prompt = f"""请将以下文本中的性别关系转换为全男性，涉及女性气质的行为动作改为符合男性气质的行为动作。使用这些角色映射：
                                {mappings_str}

                                背景上下文：
                                {context}

                                需要转换的文本：
                                {current_chapter['content']}

                                请直接返回转换后的markdown文本内容，不需要其他解释。"""
        logging.info(transform_prompt[-500:])

        # 4. 调用LLM进行转换
        logging.info(f'正在转换章节 {chapter_id}: {current_chapter["title"]}')
        try:
            transformed_content = self._call_openai(transform_prompt, temperature=0.8)
            logging.info(transformed_content[-500:])
            self.chapters_df.at[chapter_id, 'processed_content'] = transformed_content
        except Exception as e:
            print(f"处理章节 {chapter_id} 时发生错误: {str(e)}")
            raise

    def process_novel(self) -> None:
        """处理整本小说"""
        if not self.load_checkpoint():
            self.load_epub()

        for idx in range(len(self.chapters_df)):
            if pd.isna(self.chapters_df.loc[idx, 'processed_content']):
                logging.info(f'处理章节 {idx}: {self.chapters_df.loc[idx, "title"]}')
                self.process_chapter(idx)
                self.save_checkpoint()

    def export_to_epub(self, output_path: str) -> None:
        """导出处理后的内容为新的EPUB文件，并保存为Markdown文件"""
        book = epub.EpubBook()
        book.set_title(f"转换后的小说")

        chapters = []
        markdown_content = ""

        for idx, row in self.chapters_df.iterrows():
            chapter = epub.EpubHtml(
                title=row['title'],
                file_name=f'chapter_{idx}.xhtml',
                content=row['processed_content']
            )
            book.add_item(chapter)
            chapters.append(chapter)
            markdown_content += f"# {row['title']}\n\n{row['processed_content']}\n\n"

        book.toc = chapters
        book.spine = ['nav'] + chapters
        epub.write_epub(output_path, book)

        # 保存为Markdown文件
        markdown_path = output_path.replace('.epub', '.md')
        with open(markdown_path, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)


def setup_logging(log_dir: str) -> None:
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'conversion.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    parser = argparse.ArgumentParser(description='小说性别转换工具')
    # parser.add_argument('input_epub', default='source/test.epub', help='输入的EPUB文件路径')
    parser.add_argument('--output-dir', default=CONFIG['OUTPUT_DIR'], help='输出目录')
    parser.add_argument('--api-key', default=CONFIG['OPENAI_API_KEY'], help='OpenAI API密钥')
    args = parser.parse_args()

    # 创建必要的目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / CONFIG['CHECKPOINT_DIR']
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = output_dir / 'logs'

    # 设置日志
    setup_logging(str(log_dir))
    # logging.info(f"开始处理文件: {args.input_epub}")
    logging.info(f'开始处理文件: source/test.epub')

    try:
        # 初始化转换器
        gayifier = NovelGayifier(
            # epub_path=args.input_epub,
            epub_path='source/test.epub',
            save_dir=str(checkpoint_dir),
            api_key=args.api_key
        )

        # 处理小说
        gayifier.process_novel()

        # 导出结果
        epub_path = 'source/test.epub'
        output_path = output_dir / f"converted_{Path(epub_path).stem}.epub"
        gayifier.export_to_epub(str(output_path))

        logging.info(f"处理完成，输出文件: {output_path}")

    except Exception as e:
        logging.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    CONFIG = {
        'OPENAI_API_KEY': '',
        'OUTPUT_DIR': 'output',
        'CHECKPOINT_DIR': 'checkpoints',
        'MODEL': 'deepseek-chat',
        'MAX_RETRIES': 3
    }
    main()
