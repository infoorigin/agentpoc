CREATE OR REPLACE VIEW v_dep_data_with_stage AS
SELECT 
  d.*,
  CASE 
    WHEN d.model_run_id = run_info.latest_run_id THEN 'latest'
    WHEN d.model_run_id = run_info.previous_run_id THEN 'previous'
  END AS run_stage
FROM dep_data d
JOIN (
    SELECT 
      MAX(model_run_id) AS latest_run_id,
      (
        SELECT model_run_id 
        FROM (
            SELECT DISTINCT model_run_id 
            FROM dep_data 
            ORDER BY model_run_id DESC 
            LIMIT 1 OFFSET 1
        ) AS prev_sub
      ) AS previous_run_id
) AS run_info
ON d.model_run_id IN (run_info.latest_run_id, run_info.previous_run_id);
