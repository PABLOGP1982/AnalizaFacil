import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import plotly.express as px
import io
import os
import itertools
import math
from joblib import Parallel, delayed

st.set_page_config(layout="wide")
st.title('Analiza Fácil')

def notnan(x, default=0):
    return x if not (isinstance(x, float) and math.isnan(x)) else default

def map_sqx_csv_to_standard(df, filename):
    rename = {
        "Open time": "Open Time", "Close time": "Close Time", "Open price": "Price",
        "Close price": "Close Price", "Symbol": "Item", "Profit/Loss": "Profit",
        "Commission": "Commission", "Swap": "Swap", "MagicNumber": "MagicNumber"
    }
    columns_renamed = {k: v for k, v in rename.items() if k in df.columns}
    df = df.rename(columns=columns_renamed)
    for col in ["S / L", "T / P", "Taxes", "Coment"]:
        if col not in df.columns:
            df[col] = "" if col == "Coment" else np.nan
    df["Cleaned_EA"] = filename
    needed = [
        "Type", "Size", "Item", "Price", "S / L", "T / P", "Close Time",
        "Close Price", "Commission", "Taxes", "Swap", "Profit",
        "MagicNumber", "Coment", "Cleaned_EA"
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = ""
    if "Ticket" not in df.columns:
        df["Ticket"] = np.arange(1, len(df) + 1)
    return df

def cargar_y_limpiar_datos(uploaded_files, force_filename=False):
    dfs = []
    columnas_necesarias = [
        "Ticket", "Open Time", "Type", "Size", "Item", "Price", "S / L", "T / P",
        "Close Time", "Close Price", "Commission", "Taxes", "Swap", "Profit",
        "MagicNumber", "Coment", "Cleaned_EA"
    ]
    tipos_forzados = {
        "Ticket": pl.Float64, "Size": pl.Float64, "Price": pl.Float64, "S / L": pl.Float64,
        "T / P": pl.Float64, "Close Price": pl.Float64, "Commission": pl.Float64,
        "Taxes": pl.Float64, "Swap": pl.Float64, "Profit": pl.Float64, "MagicNumber": pl.Float64,
    }
    for uf in uploaded_files:
        try:
            is_csv = uf.name.lower().endswith('.csv')
            is_excel = uf.name.lower().endswith('.xlsx') or uf.name.lower().endswith('.xls')
            if is_csv:
                uf.seek(0)
                content = uf.read()
                uf.seek(0)
                try:
                    content_str = content.decode("utf-8-sig")
                except UnicodeDecodeError:
                    content_str = content.decode("latin1")
                head = content_str.split('\n',1)[0]
                delimiter = ';' if head.count(';') > head.count(',') else ','
                df_part = pd.read_csv(io.StringIO(content_str), delimiter=delimiter)
                if force_filename:
                    fname_noext = os.path.splitext(os.path.basename(uf.name))[0]
                    df_part = map_sqx_csv_to_standard(df_part, fname_noext)
            elif is_excel:
                df_part = pd.read_excel(uf, engine='openpyxl')
            else:
                st.error(f"Solo se aceptan archivos CSV o Excel. ({uf.name})")
                return None
            if "Price.1" in df_part.columns:
                df_part = df_part.rename(columns={"Price.1": "Close Price"})
            price_locs = [i for i, c in enumerate(df_part.columns) if c == "Price"]
            if len(price_locs) > 1:
                cols = list(df_part.columns)
                cols[price_locs[1]] = "Close Price"
                df_part.columns = cols
            if (not force_filename) and "Cleaned_EA" not in df_part.columns:
                posibles = [c for c in df_part.columns if c.lower() in ['coment','comment','ea','magic','magicnumber']]
                si_ea = 'Coment' if 'Coment' in df_part.columns else (posibles[0] if posibles else None)
                if si_ea:
                    df_part["Cleaned_EA"] = df_part[si_ea].astype(str)
                else:
                    df_part["Cleaned_EA"] = "EA_desconocido"
            df_part = pl.from_pandas(df_part)
            for col, tpo in tipos_forzados.items():
                if col in df_part.columns:
                    df_part = df_part.with_columns(
                        [pl.col(col).cast(tpo, strict=False).alias(col)])
            falta_col = [col for col in columnas_necesarias if col not in df_part.columns]
            if falta_col:
                st.error(f"El archivo {uf.name} no tiene todas las columnas requeridas. Faltan: {falta_col}")
                return None
            columnas_finales = [
                "Ticket", "Open Time", "Type", "Size", "Item", "Price", "S / L", "T / P",
                "Close Time", "Close Price", "Commission", "Taxes", "Swap", "Profit",
                "MagicNumber", "Coment", "Cleaned_EA"
            ]
            for col in [c for c in columnas_finales if c not in df_part.columns]:
                df_part = df_part.with_columns([pl.lit(np.nan).alias(col)])
            df_part = df_part.select(columnas_finales)
            dfs.append(df_part)
        except Exception as e:
            st.error(f"No se pudo leer el archivo {uf.name}. Error: {e}")
            return None
    if dfs:
        return pl.concat(dfs)
    else:
        st.error("Ningún archivo válido se pudo cargar.")
        return None

uploaded_mt4 = st.file_uploader(
    "Archivos de MT4/cTrader (multi-EA)",
    type=['csv','xlsx','xls'],
    accept_multiple_files=True,
    key="mt4"
)
uploaded_sqx = st.file_uploader(
    "Archivos CSV/XLSX de StrategyQuant O EXPORTADOS DE LA APP (mono-EA/nombre del archivo)",
    type=['csv','xlsx','xls'],
    accept_multiple_files=True,
    key="sqx"
)
df_mt4 = cargar_y_limpiar_datos(uploaded_mt4, force_filename=False) if uploaded_mt4 else None
df_sqx = cargar_y_limpiar_datos(uploaded_sqx, force_filename=True) if uploaded_sqx else None
df = None
if df_mt4 is not None and df_sqx is not None:
    df = pl.concat([df_mt4, df_sqx])
elif df_mt4 is not None:
    df = df_mt4
elif df_sqx is not None:
    df = df_sqx

if df is not None and not df.is_empty():
    tab_dashboard, tab_optim, tab_auto_optim = st.tabs(['📊 Dashboard', '🔨 Optimizador de Portafolios', '🤖 Optimizador Automático'])

    with tab_dashboard:
        with pl.StringCache():
            df = df.with_columns([
                pl.col('Open Time').cast(pl.Utf8).str.strptime(pl.Datetime, format="%Y.%m.%d %H:%M:%S", strict=False).alias('Open Time'),
                pl.col('Close Time').cast(pl.Utf8).str.strptime(pl.Datetime, format="%Y.%m.%d %H:%M:%S", strict=False).alias('Close Time'),
                pl.col('Cleaned_EA').cast(pl.Utf8).alias('Cleaned_EA'),
            ])
            df = df.with_columns([pl.col('Close Time').dt.strftime('%Y-%m').alias('Mes')])
            excluidos = ["deposited","deposit","withdraw","withdrawal","api","apf","transfe","canceled","cancelled","cancelado"]
            df = df.filter(
                pl.col('Coment').is_not_null() &
                ~pl.col('Coment').str.to_lowercase().str.contains('|'.join(excluidos))
            )

        st.sidebar.header("Filtros")
        fecha_min_raw = df['Close Time'].min()
        fecha_max_raw = df['Close Time'].max()
        fecha_min = pd.Timestamp(fecha_min_raw).date() if fecha_min_raw else None
        fecha_max = pd.Timestamp(fecha_max_raw).date() if fecha_max_raw else None
        fecha_desde, fecha_hasta = st.sidebar.date_input(
            'Rango de fechas (Close Time)',
            value=[fecha_min, fecha_max],
            min_value=fecha_min, max_value=fecha_max,
            key="main_rango_fechas"
        )
        if isinstance(fecha_desde, list):
            fecha_desde, fecha_hasta = fecha_desde
        multiplicador = st.sidebar.number_input("Multiplicador", value=1.0, min_value=0.0, step=0.1)
        filtro_texto = st.sidebar.text_input("Buscar texto en nombre de EA:", "")
        profit_min = st.sidebar.number_input("Profit total mínimo", value=-99999.0)
        dd_maximo_max = st.sidebar.number_input("DD máximo absoluto límite", value=9999999.0)
        mask_fechas = (df['Close Time'].dt.date() >= fecha_desde) & (df['Close Time'].dt.date() <= fecha_hasta)
        df_f = df.filter(mask_fechas)
        if filtro_texto.strip():
            terminos = [t.strip() for t in filtro_texto.replace(",", ";").split(";") if t.strip()]
            if terminos:
                mask = df_f.select(
                    pl.col('Cleaned_EA').str.strip_chars().str.to_lowercase().map_elements(
                        lambda x: any(t.lower() in x for t in terminos), return_dtype=pl.Boolean
                    ).alias('filtro')
                )['filtro']
                df_f = df_f.filter(mask)
        if df_f.is_empty():
            st.warning("No hay datos que coincidan con el filtro actual.")
            st.stop()
        def calcular_drawdown_maximo_curva(series_profit):
            valores = np.array(series_profit)
            if valores.size == 0:
                return np.nan
            saldo = valores.cumsum()
            peak = np.maximum.accumulate(saldo)
            dd = saldo - peak
            return dd.min() if dd.size > 0 else np.nan
        def calcular_resumen(df, multiplicador=1.0):
            df = df.with_columns([
                (pl.col('Profit') * multiplicador).alias('Profit_neto'),
                pl.col('Close Time').dt.strftime('%Y-%m').alias('Mes')
            ])
            meses_disponibles = df.select('Mes').unique().sort('Mes')['Mes'].to_list()
            resumen_lista = []
            eas_unicos = df.select('Cleaned_EA').unique()['Cleaned_EA'].to_list()
            for ea in eas_unicos:
                dfg = df.filter(pl.col('Cleaned_EA') == ea)
                if dfg.height == 0:
                    continue
                profit_total = dfg['Profit_neto'].sum()
                dd_max = calcular_drawdown_maximo_curva(dfg['Profit_neto'].to_numpy())
                trades = dfg.height
                grupos_df = dfg.group_by('Mes').agg(pl.col('Profit_neto').sum().alias('Profit_mes')) if 'Mes' in dfg.columns and dfg.height > 0 else None
                profit_por_mes = dict(zip(grupos_df['Mes'], grupos_df['Profit_mes'])) if grupos_df is not None else {}
                profit_positivo = dfg.filter(pl.col('Profit_neto') > 0)['Profit_neto'].sum()
                profit_negativo = dfg.filter(pl.col('Profit_neto') < 0)['Profit_neto'].sum()
                profit_factor = profit_positivo / abs(profit_negativo) if profit_negativo != 0 else np.nan
                win_ratio = dfg.filter(pl.col('Profit_neto') > 0).height / trades if trades > 0 else 0
                avg_trade = dfg['Profit_neto'].mean() if trades > 0 else 0
                ret_dd = profit_total / abs(dd_max) if dd_max not in (0, np.nan, None) and not np.isnan(dd_max) else np.nan
                fila = {
                    'Cleaned_EA': ea,
                    'Profit_Total': profit_total,
                    'Ret/DD': ret_dd,
                    'DD_Max_Curva': dd_max,
                    'Trades': trades,
                    'Profit_Factor': profit_factor,
                    'Win_Ratio_%': 100 * win_ratio,
                    'Avg_Profit_Trade': avg_trade
                }
                for mes in meses_disponibles:
                    fila[f'Profit_{mes}'] = profit_por_mes.get(mes, 0)
                resumen_lista.append(fila)
            return pl.DataFrame(resumen_lista), meses_disponibles
        resumen, meses_disponibles = calcular_resumen(df_f, multiplicador)
        resumen = resumen.filter(
            (pl.col('Profit_Total') >= profit_min) & (pl.col('DD_Max_Curva') >= -dd_maximo_max)
        )
        st.markdown("---")
        st.markdown("#### DD máximo global (portafolio todas las EAs seleccionadas combinadas, en el periodo)")
        df_portafolio = df_f.filter(pl.col('Cleaned_EA').is_in(resumen['Cleaned_EA'])).sort('Close Time')
        df_portafolio = df_portafolio.with_columns(
            (pl.col('Profit') * multiplicador).alias('Profit_neto')
        ).with_columns(pl.col('Profit_neto').cum_sum().alias('Saldo_Acumulado'))
        saldo = df_portafolio['Saldo_Acumulado'].to_numpy()
        peak = np.maximum.accumulate(saldo)
        dd = saldo - peak
        dd_max_portafolio = dd.min() if dd.size > 0 else np.nan
        st.metric("DD máximo global (portafolio)", f"{dd_max_portafolio:,.2f}")
        def mostrar_kpis(resumen, dd_max_portafolio, df_portafolio, multiplicador):
            total_profit = (df_portafolio['Profit'] * multiplicador).sum()
            profit_total_neto = total_profit
            total_trades = len(df_portafolio)
            win_ratio = resumen['Win_Ratio_%'].to_numpy()
            trades_ea = resumen['Trades'].to_numpy()
            win_ratio_avg = np.average(win_ratio, weights=trades_ea) if total_trades > 0 else 0
            ret_dd_global = profit_total_neto / abs(dd_max_portafolio) if dd_max_portafolio != 0 else np.nan
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Profit total neto", f"{profit_total_neto:,.2f}")
            kpi2.metric("Total Trades", int(total_trades))
            kpi3.metric("Win Ratio Promedio (%)", f"{win_ratio_avg:,.1f}")
            kpi4.metric("Ret/DD Global", f"{ret_dd_global:,.2f}" if not np.isnan(ret_dd_global) else "N/A")
        mostrar_kpis(resumen, dd_max_portafolio, df_portafolio, multiplicador)
        fig = px.line(df_portafolio.to_pandas(), x='Close Time', y='Saldo_Acumulado', title="Curva de saldo combinada seleccionada")
        st.plotly_chart(fig, use_container_width=True)
        figdd = px.line(df_portafolio.to_pandas(), x='Close Time', y=dd, title="Drawdown global en el tiempo")
        st.plotly_chart(figdd, use_container_width=True)
        st.dataframe(resumen.to_pandas().set_index('Cleaned_EA'), use_container_width=True)

        # =========  DESCARGAS EXCEL ============
        import time

        # 1. Descargar resumen
        def descargar_dataframe(df, nombre_archivo, label):
            towrite = io.BytesIO()
            df.to_pandas().to_excel(towrite, index=False, engine='openpyxl')
            towrite.seek(0)
            st.download_button(
                label=label,
                data=towrite,
                file_name=nombre_archivo,
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key=f"{label}_{time.time()}"
            )

        descargar_dataframe(resumen, "resumen_EAs.xlsx", "Exce: tabla resumen")

        # 2. Descargar todos los trades en formato estándar + info pegada
        columnas_estandar = [
            "Ticket", "Open Time", "Type", "Size", "Item", "Price", "S / L", "T / P",
            "Close Time", "Close Price", "Commission", "Taxes", "Swap", "Profit",
            "MagicNumber", "Coment", "Cleaned_EA"
        ]
        for col in columnas_estandar:
            if col not in df_f.columns:
                df_f = df_f.with_columns([pl.lit("").alias(col)])
        df_f_export = df_f.select(columnas_estandar)
        df_f_export = df_f_export.with_columns([
            pl.when(pl.col("Coment").is_null() | (pl.col("Coment") == ""))
            .then(pl.col("Cleaned_EA")).otherwise(pl.col("Coment")).alias("Coment"),
        ])
        df_f_export = df_f_export.with_columns([
            pl.col('Open Time').dt.strftime('%Y.%m.%d %H:%M:%S').alias('Open Time'),
            pl.col('Close Time').dt.strftime('%Y.%m.%d %H:%M:%S').alias('Close Time'),
        ])
        col_btn, col_info = st.columns([2, 10], gap="small")
        with col_btn:
            descargar_dataframe(df_f_export, "Export_trades_EAs.xlsx", "Excel: Todos Trades")
        with col_info:
            st.markdown(
                '<div style="margin-top:8px; margin-left:-32px">'
                '<span style="font-size:16px;">'
                '<span style="color:#2a8cff;"><b>ℹ️</b></span> '
                'Este excel sirve de entrada para este y otros programas'
                '</span></div>',
                unsafe_allow_html=True
            )

        # 3. Descargar excel mensual por hoja
        def generar_excel_mensual(df_f, multiplicador):
            try:
                if df_f.is_empty():
                    st.warning("No hay datos para generar el reporte mensual")
                    return None
                output = io.BytesIO()
                df_f = df_f.with_columns(
                    (pl.col('Profit') * multiplicador).alias('Profit_neto'),
                    pl.col('Coment').str.to_lowercase().str.contains("tp").alias('Es_TP')
                )
                meses = df_f['Mes'].unique().sort().to_list()
                if not meses:
                    st.warning("No se encontraron meses con datos")
                    return None
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    hojas_creadas = 0
                    for mes in meses:
                        try:
                            df_mes = df_f.filter(pl.col('Mes') == mes)
                            if df_mes.is_empty():
                                continue
                            resumen_mes = df_mes.group_by(['Cleaned_EA', 'Item', 'MagicNumber']).agg([
                                pl.col('Profit_neto').sum().alias('Sumatorio'),
                                pl.col('Ticket').count().alias('N_Trades'),
                                pl.col('Es_TP').sum().alias('N_TPs')
                            ])
                            resumen_mes = resumen_mes.select([
                                'Cleaned_EA', 'Item', 'Sumatorio', 'N_Trades', 'MagicNumber', 'N_TPs'
                            ])
                            resumen_mes.to_pandas().to_excel(writer, sheet_name=str(mes), index=False)
                            hojas_creadas += 1
                        except Exception as e:
                            st.error(f"Error procesando mes {mes}: {str(e)}")
                            continue
                    if hojas_creadas == 0:
                        pd.DataFrame(
                            columns=['Cleaned_EA', 'Item', 'Sumatorio', 'N_Trades', 'MagicNumber', 'N_TPs']).to_excel(
                            writer, sheet_name="Sin datos", index=False)
                        st.warning("No se encontraron datos válidos para ningún mes")
                output.seek(0)
                return output if hojas_creadas > 0 else None
            except Exception as e:
                st.error(f"Error generando Excel mensual: {str(e)}")
                return None

        excel_mensual = generar_excel_mensual(df_f, multiplicador)
        if excel_mensual is not None:
            st.download_button(
                label="Excel: Información mensual resumida",
                data=excel_mensual,
                file_name="resumen_mensual_EAs.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key=f"mensual_{time.time()}"
            )
        else:
            st.warning("No hay datos suficientes para generar el reporte mensual")

    # ==== TAB OPTIMIZADOR "TRADICIONAL" ====
    with tab_optim:
        st.header("🔨 Optimizador de Portafolios")
        estrategias_unicas = df['Cleaned_EA'].unique().to_list()
        if len(estrategias_unicas) < 2:
            st.info("Se necesitan al menos dos EAs distintos para optimizar portafolios.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                portfolio_size = st.slider("Nº de EAs en el Portafolio", 2, min(20, len(estrategias_unicas)), 4, 1)
            with c2:
                max_combos = st.number_input("Límite de combinaciones a probar", 10, 5000, 300)
            with c3:
                top_n = st.number_input("Nº portafolios óptimos a mostrar", 1, 10, 1)
            metric_option = st.selectbox(
                "Métrica de optimización", [
                    "Menor Max DD", "Mayor Profit", "Mayor Profit/DD",
                    "Más Estable (std. mensual)", "Recuperación más Rápida (max DD duration)"
                ], index=0
            )
            st.info("Pulsa para lanzar el optimizador")
            run_opt = st.button("🔍 Encontrar portafolio óptimo")
            if run_opt:
                all_combos = list(itertools.combinations(estrategias_unicas, portfolio_size))
                total_combos = len(all_combos)
                if total_combos > max_combos:
                    st.warning(f"Hay {total_combos:,} combinaciones posibles. Se probará una muestra aleatoria de {max_combos:,}.")
                    combos_to_test = [all_combos[i] for i in np.random.choice(total_combos, max_combos, replace=False)]
                else:
                    combos_to_test = all_combos
                resultados = []
                progress_bar = st.progress(0)
                for ii, combo in enumerate(combos_to_test):
                    dfg = df.filter(pl.col("Cleaned_EA").is_in(combo)).sort("Close Time")
                    dfp = dfg.to_pandas().sort_values("Close Time")
                    saldo = dfp['Profit'].cumsum().values if not dfp.empty else np.zeros(1)
                    hwm = np.maximum.accumulate(saldo)
                    dd = saldo - hwm
                    max_dd = dd.min() if len(dd) > 0 else 0
                    max_dd_idx = dd.argmin() if len(dd) > 0 else 0
                    max_dd_start = hwm[:max_dd_idx + 1].argmax() if len(hwm) > 0 else 0
                    try:
                        recovery = np.where(saldo[max_dd_idx + 1:] >= hwm[max_dd_idx])[0]
                        dd_recovery_idx = (recovery[0] + max_dd_idx + 1) if len(recovery) > 0 else len(saldo) - 1
                    except:
                        dd_recovery_idx = len(saldo) - 1
                    max_dd_duration = dd_recovery_idx - max_dd_start if dd_recovery_idx is not None and dd_recovery_idx > max_dd_start else 0
                    profit_total = float(dfp['Profit'].sum())
                    if not dfp.empty and 'Close Time' in dfp.columns:
                        dfp['mes'] = pd.to_datetime(dfp['Close Time']).dt.to_period('M')
                        profits_mensual = dfp.groupby('mes')['Profit'].sum()
                        std_month = float(profits_mensual.std()) if len(profits_mensual) >= 2 else 0.0
                    else:
                        std_month = 0.0
                    ret_dd = profit_total / abs(max_dd) if max_dd != 0 else 0
                    resultados.append({
                        "combo": combo,
                        "max_dd": max_dd,
                        "profit": profit_total,
                        "std_month": std_month,
                        "max_dd_duration": max_dd_duration,
                        "ret_dd": ret_dd
                    })
                    if (ii + 1) % 10 == 0 or (ii + 1) == len(combos_to_test):
                        progress_bar.progress((ii + 1) / len(combos_to_test))
                if metric_option == "Menor Max DD":
                    resultados_ordenados = sorted(resultados, key=lambda x: x["max_dd"], reverse=True)
                elif metric_option == "Mayor Profit":
                    resultados_ordenados = sorted(resultados, key=lambda x: x["profit"], reverse=True)
                elif metric_option == "Mayor Profit/DD":
                    resultados_ordenados = sorted(resultados, key=lambda x: x["ret_dd"], reverse=True)
                elif metric_option == "Más Estable (std. mensual)":
                    resultados_ordenados = sorted(resultados, key=lambda x: x["std_month"])
                elif metric_option == "Recuperación más Rápida (max DD duration)":
                    resultados_ordenados = sorted(resultados, key=lambda x: x["max_dd_duration"])
                else:
                    resultados_ordenados = resultados
                st.success("¡Optimización completada!")
                for idx, res in enumerate(resultados_ordenados[:top_n]):
                    st.markdown(f"### 🏆 Portafolio óptimo #{idx+1}")
                    st.write("- **EAs:**", ", ".join(res["combo"]))
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Max DD", f"{res['max_dd']:.2f}")
                    c2.metric("Profit total", f"{res['profit']:.2f}")
                    c3.metric("Std. mensual", f"{res['std_month']:.2f}")
                    c4.metric("Duración DD", f"{res['max_dd_duration']}")
                    st.write(f"**Profit/DD:** {res['ret_dd']:.2f}")
                    dfg = df.filter(pl.col("Cleaned_EA").is_in(res["combo"])).sort("Close Time")
                    dfp = dfg.to_pandas().sort_values("Close Time")
                    dfp['Saldo_Acumulado'] = dfp['Profit'].cumsum()
                    towrite = io.BytesIO()
                    dfp.to_excel(towrite, index=False, engine='openpyxl')
                    towrite.seek(0)
                    st.download_button(
                        label=f"Descargar trades de portafolio óptimo #{idx+1} en Excel",
                        data=towrite,
                        file_name=f"portafolio_optimo_{idx+1}.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        key=f"download_opt_resum_{idx+1}"
                        )
                    if len(resultados_ordenados) > 0:
                        n_eas = portfolio_size
                        ea_cols = [f"EA_{i + 1}" for i in range(n_eas)]
                        tabla = []
                        for idx, res in enumerate(resultados_ordenados[:top_n]):
                            fila = [idx + 1]
                            eas = list(res["combo"])
                            eas += [None] * (n_eas - len(eas))
                            fila.extend(eas)
                            fila.extend([
                                int(round(res['max_dd'])),
                                int(round(res['profit'])),
                                int(round(res['std_month'])),
                                int(round(res['max_dd_duration'])),
                                round(res['ret_dd'], 2)
                            ])
                            tabla.append(fila)
                        df_out = pd.DataFrame(
                            tabla,
                            columns=(["Rank"] + ea_cols + ["Max DD", "Profit total", "Estabilidad (Std)", "Duración DD",
                                                           "Profit/DD"])
                        )
                        excel_bytes = io.BytesIO()
                        with pd.ExcelWriter(excel_bytes, engine='xlsxwriter') as writer:
                            workbook = writer.book
                            worksheet = workbook.add_worksheet("Resumen")
                            writer.sheets["Resumen"] = worksheet
                            header_fmt = workbook.add_format(
                                {'bold': True, 'bg_color': '#D9E1F2', 'align': 'center', 'border': 1})
                            for ci, col in enumerate(df_out.columns):
                                worksheet.write(0, ci, col, header_fmt)
                            for ri, fila in enumerate(df_out.values):
                                for ci, val in enumerate(fila):
                                    worksheet.write(ri + 1, ci, val)
                            integer_format = workbook.add_format({'num_format': '0'})
                            float_format = workbook.add_format({'num_format': '0.00'})
                            for col in range(len(["Rank"] + ea_cols), len(df_out.columns) - 1):
                                worksheet.set_column(col, col, 15, integer_format)
                            worksheet.set_column(len(df_out.columns) - 1, len(df_out.columns) - 1, 13, float_format)
                        excel_bytes.seek(0)
                        import time
                        st.download_button(
                            label="⬇️ Descargar Excel RESUMEN Ranking",
                            data=excel_bytes,
                            file_name="optimizador_resumen.xlsx",
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            key=f"download_traditional_summary_{time.time()}"
                        )
    # ==== TAB AUTO-OPTIMIZADOR TOTALMENTE OPTIMIZADO ====
    with tab_auto_optim:
        if 'auto_metrics' not in st.session_state:
            st.session_state['auto_metrics'] = ["Menor Max DD", "Mayor Profit"]
        if 'auto_nums' not in st.session_state:
            st.session_state['auto_nums'] = "5,10"
        if 'auto_limit' not in st.session_state:
            st.session_state['auto_limit'] = 1000
        if 'auto_ntop' not in st.session_state:
            st.session_state['auto_ntop'] = 5
        metricas = [
            "Menor Max DD", "Mayor Profit", "Mayor Profit/DD",
            "Más Estable (std. mensual)", "Recuperación más Rápida (max DD duration)"
        ]
        st.multiselect("Métricas de optimización", metricas, key="auto_metrics")
        st.text_input("Nº de EAs en el Portafolio (separa con comas)", key="auto_nums")
        st.number_input("Límite de combinaciones por cada tamaño de portafolio", 10, 10000, key="auto_limit")
        st.number_input("Nº portafolios óptimos a mostrar por tamaño/métrica", 1, 50, key="auto_ntop")
        metricas_sel = st.session_state['auto_metrics']
        nums_eas_in = st.session_state['auto_nums']
        limite_combos = st.session_state['auto_limit']
        n_top_port = st.session_state['auto_ntop']
        try:
            lista_n_eas = sorted(set(int(x.strip()) for x in nums_eas_in.split(",") if x.strip().isdigit() and int(x.strip()) >= 2))
            if not lista_n_eas:
                lista_n_eas = [5, 10]
        except:
            lista_n_eas = [5, 10]
        boton_auto = st.button("🔍 Encontrar Portafolios Óptimos", key="button_auto_optim")
        estado = st.empty()
        progress = st.progress(0)
        if boton_auto:
            estrategias_unicas = df['Cleaned_EA'].unique().to_list()
            df_all = df.to_pandas().copy().sort_values("Close Time")
            resultados_por_bloque = {}
            total_tareas = len(lista_n_eas) * len(metricas_sel)
            paso_actual = 0
            saldo_por_ea = {}
            mesprofit_por_ea = {}
            for ea in estrategias_unicas:
                sdf = df_all[df_all['Cleaned_EA'] == ea].sort_values("Close Time")
                saldo_por_ea[ea] = sdf['Profit'].cumsum().reset_index(drop=True).values
                if not sdf.empty and 'Close Time' in sdf.columns:
                    meses = pd.to_datetime(sdf['Close Time']).dt.to_period('M')
                    mesprofit_por_ea[ea] = sdf.groupby(meses)['Profit'].sum()
                else:
                    mesprofit_por_ea[ea] = pd.Series(dtype=float)
            def calc_metrics(combo):
                maxlen = max(len(saldo_por_ea[ea]) for ea in combo)
                curvas = []
                for ea in combo:
                    curva = saldo_por_ea[ea]
                    if len(curva) < maxlen:
                        fill_val = curva[-1] if len(curva) > 0 else 0
                        curva = np.concatenate([curva, np.full(maxlen - len(curva), fill_val)])
                    curvas.append(curva)
                arr = np.sum(curvas, axis=0)
                hwm = np.maximum.accumulate(arr)
                dd = arr - hwm
                max_dd = dd.min() if len(dd) > 0 else 0
                max_dd_idx = dd.argmin() if len(dd) > 0 else 0
                max_dd_start = hwm[:max_dd_idx + 1].argmax() if len(hwm) > 0 else 0
                try:
                    recovery = np.where(arr[max_dd_idx + 1:] >= hwm[max_dd_idx])[0]
                    dd_recovery_idx = (recovery[0] + max_dd_idx + 1) if len(recovery) > 0 else len(arr) - 1
                except:
                    dd_recovery_idx = len(arr) - 1
                max_dd_duration = dd_recovery_idx - max_dd_start if dd_recovery_idx is not None and dd_recovery_idx > max_dd_start else 0
                profit_total = float(arr[-1]) if len(arr) > 0 else 0
                comb_mes = sum((mesprofit_por_ea[ea] for ea in combo), pd.Series(dtype=float))
                std_month = float(comb_mes.std()) if len(comb_mes) >= 2 else 0.0
                ret_dd = profit_total / abs(max_dd) if max_dd != 0 else 0
                return {
                    "Rank": 0,
                    **{f"EA_{i+1}": ea for i,ea in enumerate(combo)},
                    "Max DD": int(round(notnan(max_dd))),
                    "Profit total": int(round(notnan(profit_total))),
                    "Estabilidad (Std)": int(round(notnan(std_month))),
                    "Duración DD": int(round(notnan(max_dd_duration))),
                    "Profit/DD": round(notnan(ret_dd),2)
                }
            for n_eas in lista_n_eas:
                from random import sample

                n_eas_actual = n_eas
                total_eas = len(estrategias_unicas)
                max_combos = math.comb(total_eas, n_eas_actual)

                if max_combos <= limite_combos:
                    # Si es razonable, sí las generamos todas
                    combos_sample = list(itertools.combinations(estrategias_unicas, n_eas_actual))
                else:
                    # Genera muestra aleatoria sin generar todas primero
                    combos_set = set()
                    while len(combos_set) < limite_combos:
                        combo = tuple(sorted(sample(estrategias_unicas, n_eas_actual)))
                        combos_set.add(combo)
                    combos_sample = list(combos_set)
                for metrica in metricas_sel:
                    estado.info(f"Nº EAs: {n_eas} | Métrica: {metrica} | Combos: {len(combos_sample)}")
                    partial_results = Parallel(n_jobs=-1, prefer="threads")(
                        delayed(calc_metrics)(combo) for combo in combos_sample
                    )
                    if metrica == "Menor Max DD":
                        partial_results.sort(key=lambda x: x["Max DD"], reverse=True)
                    elif metrica == "Mayor Profit":
                        partial_results.sort(key=lambda x: x["Profit total"], reverse=True)
                    elif metrica == "Mayor Profit/DD":
                        partial_results.sort(key=lambda x: x["Profit/DD"], reverse=True)
                    elif metrica == "Más Estable (std. mensual)":
                        partial_results.sort(key=lambda x: x["Estabilidad (Std)"])
                    elif metrica == "Recuperación más Rápida (max DD duration)":
                        partial_results.sort(key=lambda x: x["Duración DD"])
                    for k, d in enumerate(partial_results[:n_top_port], start=1):
                        d['Rank'] = k
                    resultados_por_bloque[(n_eas, metrica)] = partial_results[:n_top_port]
                    paso_actual += 1
                    progress.progress(paso_actual / total_tareas)
            estado.success("¡Optimización completada! Generando Excel...")
            import xlsxwriter
            excel_out = io.BytesIO()
            with pd.ExcelWriter(excel_out, engine='xlsxwriter') as writer:
                workbook = writer.book
                worksheet = workbook.add_worksheet("Portafolios_Optimos")
                writer.sheets["Portafolios_Optimos"] = worksheet

                header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'align': 'center', 'border': 1})
                title_fmt = workbook.add_format({'bold': True, 'align': 'center'})
                col_offset = 0
                for n_eas in lista_n_eas:
                    local_col = col_offset
                    row_offset = 0
                    for metrica in metricas_sel:
                        resultados = resultados_por_bloque.get((n_eas, metrica), [])
                        if not resultados:
                            continue
                        columnas = ["Rank"] + [f"EA_{i+1}" for i in range(n_eas)] + [
                            "Max DD", "Profit total", "Estabilidad (Std)", "Duración DD", "Profit/DD"
                        ]
                        df_out = pd.DataFrame(resultados, columns=columnas)
                        title_row = row_offset
                        worksheet.write(title_row, local_col, f"{n_eas} eas", title_fmt)
                        worksheet.write(title_row, local_col + 1, metrica, title_fmt)

                        for ci, colname in enumerate(columnas):
                            worksheet.write(title_row + 1, local_col + ci, colname, header_fmt)
                        for ri, row in enumerate(df_out.values):
                            for ci, val in enumerate(row):
                                worksheet.write(title_row+2+ri, local_col+ci, val)
                        if n_eas >= 3:
                            ea_cols_first = local_col+1
                            ea_cols_last = local_col+n_eas
                            worksheet.set_column(ea_cols_first, ea_cols_last, 12, None, {'level': 1, 'hidden': 1})
                        worksheet.set_column(local_col, local_col, 7)
                        for ci in range(n_eas):
                            worksheet.set_column(local_col+1+ci, local_col+1+ci, 15)
                        worksheet.set_column(local_col+n_eas+1, local_col+n_eas+5, 14)
                        worksheet.set_column(local_col+n_eas+6, local_col+n_eas+6, 2)
                        block_len = len(df_out)+4
                        row_offset += block_len
                    row_offset = 0
                    col_offset += n_eas+7
            excel_out.seek(0)
            st.session_state['excel_auto_result'] = excel_out.getvalue()
            st.balloons()
            st.markdown("#### ⬇️ DESCARGA TU EXCEL AQUÍ:")
            st.download_button(
                "⬇️ Descargar Excel con TODOS los portafolios óptimos",
                data=st.session_state['excel_auto_result'],
                file_name="optimizador_automatico.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key="download_optim_automatico_excel"
            )
            st.markdown("""
                <script>
                    setTimeout(() => { window.scrollTo(0, document.body.scrollHeight); }, 600);
                </script>
            """, unsafe_allow_html=True)
            progress.empty()
            estado.success("¡Archivo Excel listo para descargar!")
else:
    st.info("Por favor, sube archivos de MT4/cTrader y/o StrategyQuant en los botones correspondientes arriba.")
