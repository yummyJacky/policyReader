"""
Word和HTML转PDF工具类

安装依赖:
pip install weasyprint

系统依赖:
- LibreOffice: 用于Word文档转换
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod


class BaseConverter(ABC):
    """转换器基类"""
    
    @abstractmethod
    def convert(self, input_path: str, output_path: str) -> bool:
        pass


class WordConverter(BaseConverter):
    """Word文档转PDF转换器"""
    
    def __init__(self):
        self.libreoffice_path = self._find_libreoffice()
    
    def _find_libreoffice(self) -> Optional[str]:
        """查找LibreOffice路径"""
        # 检查PATH
        for cmd in ['libreoffice', 'soffice']:
            if shutil.which(cmd):
                return cmd
        
        # 检查常见安装路径
        paths = [
            '/usr/bin/libreoffice',
            '/usr/bin/soffice',
            '/Applications/LibreOffice.app/Contents/MacOS/soffice',
            r'C:\Program Files\LibreOffice\program\soffice.exe',
            r'C:\Program Files (x86)\LibreOffice\program\soffice.exe',
        ]
        
        for path in paths:
            if os.path.exists(path):
                return path
        return None
    
    def convert(self, input_path: str, output_path: str) -> bool:
        # 方式1: LibreOffice (跨平台，推荐)
        if self.libreoffice_path:
            return self._convert_with_libreoffice(input_path, output_path)
        
        # 方式2: docx2pdf (仅Windows)
        try:
            return self._convert_with_docx2pdf(input_path, output_path)
        except Exception:
            pass
        
        print("错误: 未找到可用的Word转换工具，请安装LibreOffice")
        return False
    
    def _convert_with_libreoffice(self, input_path: str, output_path: str) -> bool:
        """使用LibreOffice转换"""
        try:
            output_dir = os.path.dirname(os.path.abspath(output_path)) or '.'
            
            cmd = [
                self.libreoffice_path,
                '--headless',
                '--convert-to', 'pdf',
                '--outdir', output_dir,
                os.path.abspath(input_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                print(f"LibreOffice错误: {result.stderr}")
                return False
            
            # 处理输出文件名
            generated_pdf = os.path.join(
                output_dir,
                Path(input_path).stem + '.pdf'
            )
            
            if generated_pdf != os.path.abspath(output_path):
                if os.path.exists(generated_pdf):
                    shutil.move(generated_pdf, output_path)
            
            return os.path.exists(output_path)
            
        except subprocess.TimeoutExpired:
            print("转换超时")
            return False
        except Exception as e:
            print(f"LibreOffice转换失败: {e}")
            return False
    
    def _convert_with_docx2pdf(self, input_path: str, output_path: str) -> bool:
        """使用docx2pdf转换 (仅Windows)"""
        from docx2pdf import convert
        convert(input_path, output_path)
        return os.path.exists(output_path)


class HTMLConverter(BaseConverter):
    """HTML/网页转PDF转换器"""
    
    def convert(self, input_path: str, output_path: str) -> bool:
        # 方式1: weasyprint (推荐)
        try:
            return self._convert_with_weasyprint(input_path, output_path)
        except ImportError:
            print("提示: 安装weasyprint可获得更好效果: pip install weasyprint")
        except Exception as e:
            print(f"weasyprint转换失败: {e}")
        
        # 方式2: pdfkit
        try:
            return self._convert_with_pdfkit(input_path, output_path)
        except ImportError:
            pass
        except Exception as e:
            print(f"pdfkit转换失败: {e}")
        
        print("错误: 未找到可用的HTML转换工具")
        return False
    
    def _convert_with_weasyprint(self, input_path: str, output_path: str) -> bool:
        """使用weasyprint转换"""
        from weasyprint import HTML
        HTML(filename=input_path).write_pdf(output_path)
        return os.path.exists(output_path)
    
    def _convert_with_pdfkit(self, input_path: str, output_path: str) -> bool:
        """使用pdfkit转换"""
        import pdfkit
        pdfkit.from_file(input_path, output_path)
        return os.path.exists(output_path)
    
    def convert_from_url(self, url: str, output_path: str) -> bool:
        """从URL转换网页为PDF"""
        # 优先使用 pdfkit + wkhtmltopdf，渲染效果更接近真实浏览器
        try:
            import pdfkit
            options = {
                "encoding": "utf-8",
                "page-size": "A4",
                "margin-top": "10mm",
                "margin-bottom": "10mm",
                "margin-left": "10mm",
                "margin-right": "10mm",
                # 可以根据需要微调缩放，避免内容被压缩过窄
                # "zoom": 1.0,
            }
            pdfkit.from_url(url, output_path, options=options)
            if os.path.exists(output_path):
                return True
        except ImportError:
            pass
        except Exception as e:
            print(f"pdfkit URL转换失败: {e}")

        # 回退到 weasyprint
        try:
            from weasyprint import HTML
            HTML(url=url).write_pdf(output_path)
            return os.path.exists(output_path)
        except ImportError:
            pass
        except Exception as e:
            print(f"weasyprint URL转换失败: {e}")

        print("错误: 未找到可用的HTML转换工具")
        return False
    
    def convert_from_string(self, html_string: str, output_path: str) -> bool:
        """从HTML字符串转换为PDF"""
        try:
            from weasyprint import HTML
            HTML(string=html_string).write_pdf(output_path)
            return os.path.exists(output_path)
        except ImportError:
            pass
        
        try:
            import pdfkit
            pdfkit.from_string(html_string, output_path)
            return os.path.exists(output_path)
        except Exception as e:
            print(f"HTML字符串转换失败: {e}")
        
        return False


class PDFConverter:
    """
    Word和HTML转PDF工具类
    
    使用示例:
        converter = PDFConverter()
        
        # Word转PDF
        converter.convert('document.docx', 'output.pdf')
        
        # HTML转PDF
        converter.convert('page.html', 'page.pdf')
        
        # URL转PDF
        converter.url_to_pdf('https://example.com', 'website.pdf')
        
        # HTML字符串转PDF
        converter.html_to_pdf('<h1>Hello</h1>', 'hello.pdf')
    """
    
    # 支持的扩展名
    WORD_EXTENSIONS = {'.doc', '.docx', '.odt', '.rtf'}
    HTML_EXTENSIONS = {'.html', '.htm', '.xhtml'}
    
    def __init__(self):
        self._word_converter = WordConverter()
        self._html_converter = HTMLConverter()
    
    def convert(
        self, 
        input_path: str, 
        output_path: Optional[str] = None
    ) -> bool:
        """
        将Word或HTML文件转换为PDF
        
        Args:
            input_path: 输入文件路径
            output_path: 输出PDF路径 (默认与输入同名)
            
        Returns:
            bool: 是否成功
        """
        if not os.path.exists(input_path):
            print(f"错误: 文件不存在 - {input_path}")
            return False
        
        if output_path is None:
            output_path = str(Path(input_path).with_suffix('.pdf'))
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        ext = Path(input_path).suffix.lower()
        
        if ext in self.WORD_EXTENSIONS:
            print(f"[Word] {input_path} -> {output_path}")
            return self._word_converter.convert(input_path, output_path)
        
        elif ext in self.HTML_EXTENSIONS:
            print(f"[HTML] {input_path} -> {output_path}")
            return self._html_converter.convert(input_path, output_path)
        
        else:
            print(f"错误: 不支持的文件类型 '{ext}'")
            print(f"支持的类型: {self.WORD_EXTENSIONS | self.HTML_EXTENSIONS}")
            return False
    
    def url_to_pdf(self, url: str, output_path: str) -> bool:
        """
        将网页URL转换为PDF
        
        Args:
            url: 网页地址
            output_path: 输出PDF路径
        """
        print(f"[URL] {url} -> {output_path}")
        return self._html_converter.convert_from_url(url, output_path)
    
    def html_to_pdf(self, html_string: str, output_path: str) -> bool:
        """
        将HTML字符串转换为PDF
        
        Args:
            html_string: HTML内容
            output_path: 输出PDF路径
        """
        print(f"[HTML String] -> {output_path}")
        return self._html_converter.convert_from_string(html_string, output_path)
    
    def batch_convert(
        self, 
        input_paths: list, 
        output_dir: Optional[str] = None
    ) -> dict:
        """
        批量转换文件
        
        Args:
            input_paths: 文件路径列表
            output_dir: 输出目录
            
        Returns:
            dict: {文件路径: 是否成功}
        """
        results = {}
        
        for input_path in input_paths:
            if output_dir:
                output_path = os.path.join(
                    output_dir, 
                    Path(input_path).stem + '.pdf'
                )
            else:
                output_path = None
            
            results[input_path] = self.convert(input_path, output_path)
        
        # 打印统计
        success = sum(results.values())
        total = len(results)
        print(f"\n转换完成: {success}/{total} 成功")
        
        return results


# ============ 使用示例 ============
if __name__ == '__main__':
    converter = PDFConverter()
    
    # 示例1: Word转PDF
    # converter.convert('policy_data/海南省农机购置与应用补贴实施方案的通知.doc', 'policy_data/海南省农机购置与应用补贴实施方案的通知.pdf')
    
    # 示例2: HTML文件转PDF
    # converter.convert('page.html', 'page.pdf')
    
    # 示例3: URL转PDF
    # https://go-lang.org.cn/doc/install
    converter.url_to_pdf('https://www.moa.gov.cn/xw/zwdt/202512/t20251205_6479513.htm', 'example_gov.pdf')
    
    # 示例4: HTML字符串转PDF
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; padding: 40px; }
            h1 { color: #333; }
            p { color: #666; line-height: 1.6; }
        </style>
    </head>
    <body>
        <h1>Hello World</h1>
        <p>这是一个HTML转PDF的示例。</p>
    </body>
    </html>
    """
    # converter.html_to_pdf(html_content, 'hello.pdf')
    
    # 示例5: 批量转换
    # files = ['doc1.docx', 'doc2.docx', 'page.html']
    # results = converter.batch_convert(files, output_dir='./pdfs/')
    
    print("工具类已加载，支持的格式:")
    print(f"  Word: {PDFConverter.WORD_EXTENSIONS}")
    print(f"  HTML: {PDFConverter.HTML_EXTENSIONS}")