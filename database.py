import json
import pymysql
from datetime import datetime
from config import DB_CONFIG, CLASS_NAMES_EN, CLASS_NAMES_CN


class Database:
    """数据库操作类，封装所有与 prediction_records 表相关的操作"""

    def __init__(self):
        self.config = DB_CONFIG.copy()
        self._ensure_database_exists()

    def _get_connection(self):
        return pymysql.connect(
            host=self.config['host'],
            user=self.config['user'],
            password=self.config['password'],
            database=self.config.get('database'),
            port=self.config.get('port', 3306),
            charset=self.config.get('charset', 'utf8mb4'),
            autocommit=False,
            connect_timeout=10,
            ssl_disabled=True
        )

    def _ensure_database_exists(self):
        temp_config = self.config.copy()
        temp_db = temp_config.pop('database', None)
        conn = pymysql.connect(
            host=temp_config['host'],
            user=temp_config['user'],
            password=temp_config['password'],
            port=temp_config.get('port', 3306),
            charset=temp_config.get('charset', 'utf8mb4'),
            ssl_disabled=True
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {temp_db} "
                               f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                conn.commit()
                print(f"数据库 '{temp_db}' 已确保存在")
        finally:
            conn.close()

    def init_table(self):
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                CREATE TABLE IF NOT EXISTS prediction_records (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    predicted_class VARCHAR(20) NOT NULL,
                    predicted_class_cn VARCHAR(50) NOT NULL,
                    confidence DECIMAL(6,4) NOT NULL,
                    all_probs JSON NOT NULL,
                    image_path VARCHAR(500) DEFAULT NULL,
                    ip_address VARCHAR(45) DEFAULT NULL,
                    INDEX idx_created_at (created_at),
                    INDEX idx_predicted_class (predicted_class)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                cursor.execute(sql)
                conn.commit()
                print("数据表 'prediction_records' 已确保存在")
        except pymysql.Error as e:
            print(f"创建表失败: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def insert_record(self, predicted_class, predicted_class_cn, confidence,
                      all_probs, image_path=None, ip_address=None):
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                all_probs_json = json.dumps(all_probs, ensure_ascii=False)
                sql = """
                INSERT INTO prediction_records 
                (predicted_class, predicted_class_cn, confidence, all_probs, 
                 image_path, ip_address)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    predicted_class, predicted_class_cn, confidence,
                    all_probs_json, image_path, ip_address
                ))
                conn.commit()
                record_id = cursor.lastrowid
                print(f"记录已插入，ID: {record_id}")
                return record_id
        except pymysql.Error as e:
            print(f"插入记录失败: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_recent_records(self, limit=20):
        conn = self._get_connection()
        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = """
                SELECT id, created_at, predicted_class, predicted_class_cn, 
                       confidence, all_probs
                FROM prediction_records
                ORDER BY created_at DESC
                LIMIT %s
                """
                cursor.execute(sql, (int(limit),))  # 确保 limit 为整数
                records = cursor.fetchall()
                for record in records:
                    if record.get('all_probs'):
                        record['all_probs'] = json.loads(record['all_probs'])
                    if record.get('created_at'):
                        record['created_at'] = record['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                return records
        except pymysql.Error as e:
            print(f"查询失败: {e}")
            raise
        finally:
            conn.close()

    def get_records_by_date(self, start_date, end_date):
        conn = self._get_connection()
        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = """
                SELECT id, created_at, predicted_class, predicted_class_cn, 
                       confidence, all_probs
                FROM prediction_records
                WHERE DATE(created_at) BETWEEN %s AND %s
                ORDER BY created_at DESC
                """
                cursor.execute(sql, (start_date, end_date))
                records = cursor.fetchall()
                for record in records:
                    if record.get('all_probs'):
                        record['all_probs'] = json.loads(record['all_probs'])
                    if record.get('created_at'):
                        record['created_at'] = record['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                return records
        except pymysql.Error as e:
            print(f"按日期查询失败: {e}")
            raise
        finally:
            conn.close()

    def get_records_by_category(self, predicted_class):
        conn = self._get_connection()
        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = """
                SELECT id, created_at, predicted_class, predicted_class_cn, 
                       confidence, all_probs
                FROM prediction_records
                WHERE predicted_class = %s
                ORDER BY created_at DESC
                """
                cursor.execute(sql, (predicted_class,))
                records = cursor.fetchall()
                for record in records:
                    if record.get('all_probs'):
                        record['all_probs'] = json.loads(record['all_probs'])
                    if record.get('created_at'):
                        record['created_at'] = record['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                return records
        except pymysql.Error as e:
            print(f"按类别查询失败: {e}")
            raise
        finally:
            conn.close()

    def get_paginated_records(self, page=1, page_size=20, category=None,
                              start_date=None, end_date=None):
        # 强制转换为整数，防止浮点数导致 SQL 语法错误
        page = int(page)
        page_size = int(page_size)
        offset = (page - 1) * page_size

        conn = self._get_connection()
        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                conditions = []
                params = []

                if category:
                    conditions.append("predicted_class = %s")
                    params.append(category)
                if start_date:
                    conditions.append("DATE(created_at) >= %s")
                    params.append(start_date)
                if end_date:
                    conditions.append("DATE(created_at) <= %s")
                    params.append(end_date)

                where_clause = " AND ".join(conditions) if conditions else "1=1"

                # 查询总数
                count_sql = f"SELECT COUNT(*) as total FROM prediction_records WHERE {where_clause}"
                cursor.execute(count_sql, params)
                total = cursor.fetchone()['total']

                total_pages = (total + page_size - 1) // page_size if total > 0 else 1

                data_sql = f"""
                SELECT id, created_at, predicted_class, predicted_class_cn, 
                       confidence, all_probs
                FROM prediction_records
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """
                cursor.execute(data_sql, params + [page_size, offset])
                records = cursor.fetchall()

                for record in records:
                    if record.get('all_probs'):
                        record['all_probs'] = json.loads(record['all_probs'])
                    if record.get('created_at'):
                        record['created_at'] = record['created_at'].strftime('%Y-%m-%d %H:%M:%S')

                return {
                    'records': records,
                    'total': total,
                    'page': page,
                    'page_size': page_size,
                    'total_pages': total_pages
                }
        except pymysql.Error as e:
            print(f"分页查询失败: {e}")
            raise
        finally:
            conn.close()

    def get_category_stats(self):
        conn = self._get_connection()
        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                sql = """
                SELECT predicted_class, COUNT(*) as count
                FROM prediction_records
                GROUP BY predicted_class
                ORDER BY count DESC
                """
                cursor.execute(sql)
                results = cursor.fetchall()
                stats = {}
                for result in results:
                    stats[result['predicted_class']] = result['count']
                for cls in CLASS_NAMES_EN:
                    if cls not in stats:
                        stats[cls] = 0
                return stats
        except pymysql.Error as e:
            print(f"统计查询失败: {e}")
            raise
        finally:
            conn.close()

    def delete_record(self, record_id):
        # 强制转为整数
        record_id = int(record_id)
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                sql = "DELETE FROM prediction_records WHERE id = %s"
                rows_affected = cursor.execute(sql, (record_id,))
                conn.commit()
                if rows_affected == 0:
                    print(f"未找到 ID={record_id} 的记录")
                    return False
                print(f"记录 ID={record_id} 已删除")
                return True
        except pymysql.Error as e:
            print(f"删除记录失败: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()


# 创建单例实例
db = Database()

if __name__ == "__main__":
    db.init_table()