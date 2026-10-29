-- Q1 WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS spend_over_500 FROM orders WHERE status='completed' AND order_amount>500 GROUP BY customer_id HAVING SUM(order_amount)>8000;
-- Q2 revenue by city/category
SELECT c.city,p.category,SUM(o.order_amount) AS revenue_inr FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY c.city,p.category ORDER BY c.city,p.category;
-- Q3 products above average catalogue price
SELECT product_id,product_name,price_inr FROM products WHERE price_inr>(SELECT AVG(price_inr) FROM products);
-- Q4 customers without completed orders
SELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed' WHERE o.order_id IS NULL;
-- Q5 CTE total spend and average spend
WITH spend AS (SELECT customer_id,SUM(order_amount) total_spend FROM orders WHERE status='completed' GROUP BY customer_id), avg_spend AS (SELECT AVG(total_spend) avg_total FROM spend) SELECT spend.* FROM spend CROSS JOIN avg_spend WHERE total_spend>avg_total;
-- Q6 ranking functions by city
WITH spend AS (SELECT c.customer_id,c.city,SUM(o.order_amount) total_spend FROM orders o JOIN customers c USING(customer_id) WHERE o.status='completed' GROUP BY c.customer_id,c.city) SELECT *,RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC,customer_id) AS row_num FROM spend;
-- Q7 running daily revenue
WITH daily AS (SELECT substr(order_ts,1,10) day,SUM(order_amount) revenue FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)) SELECT day,revenue,SUM(revenue) OVER(ORDER BY day ROWS UNBOUNDED PRECEDING) cumulative_revenue FROM daily ORDER BY day;
