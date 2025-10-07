import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import plotly.express as px
import io
import os
import itertools
import math

st.set_page_config(layout="wide")
st.title('Analiza Fácil')

# ================= FUNCIONES AUXILIARES ==================

def map_sqx_csv_to_standard(df, filename):
    rename_dict = {
        "Open time": "Open Time",
        "Close time": "Close Time",
        "Open price": "Price",
        "Close price": "Close Price",
        "Symbol": "Item",
        "Profit/Loss": "Profit",
        "Commission": "Commission",
        "Swap": "Swap",
        "MagicNumber": "MagicNumber",
    }
    columns_renamed = {k: v for k, v in rename_dict.items() if k in df.columns}
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
        "Ticket": pl.Float64,
        "Size": pl.Float64,
        "Price": pl.Float64,
        "S / L": pl.Float64,
        "T / P": pl.Float64,
        "Close Price": pl.Float64,
        "Commission": pl.Float64,
        "Taxes": pl.Float64,
        "Swap": pl.Float64,
        "Profit": pl.Float64,
        "MagicNumber": pl.Float64,
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
                        [pl.col(col).cast(tpo, strict=False).alias(col)]
                    )
            falta_col = [col for col in columnas_necesarias if col not in df_part.columns]
            if falta_col:
                st.error(f"El archivo {uf.name} no tiene todas las columnas requeridas. "
                         f"Las que faltan: {falta_col}")
                return None
            columnas_finales = [
                "Ticket", "Open Time", "Type", "Size", "Item", "Price", "S / L", "T / P",
                "Close Time", "Close Price", "Commission", "Taxes", "Swap", "Profit",
                "MagicNumber", "Coment", "Cleaned_EA"
            ]
            faltan = [col for col in columnas_finales if col not in df_part.columns]
            for col in faltan:
                df_part = df_part.with_columns([pl.lit(np.nan).alias(col)])
            df_part = df_part.select(columnas_finales)
            dfs.append(df_part)
        except Exception as e:
            st.error(f"No se pudo leer el archivo {uf.name}. Error: {e}")
            return None
    if dfs:
        df = pl.concat(dfs)
        return df
    else:
        st.error("Ningún archivo válido se pudo cargar.")
        return None

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
        if 'Mes' in dfg.columns and dfg.height > 0:
            grupos_df = dfg.group_by('Mes').agg(pl.col('Profit_neto').sum().alias('Profit_mes'))
            grupos = grupos_df.to_dict(as_series=False)
            profit_por_mes = dict(zip(grupos['Mes'], grupos['Profit_mes']))
        else:
            profit_por_mes = {}
        profit_positivo = dfg.filter(pl.col('Profit_neto') > 0)['Profit_neto'].sum()
        profit_negativo = dfg.filter(pl.col('Profit_neto') < 0)['Profit_neto'].sum()
        profit_factor = profit_positivo / abs(profit_negativo) if profit_negativo != 0 else np.nan
        win_ratio = dfg.filter(pl.col('Profit_neto') > 0).height / trades if trades > 0 else 0
        avg_trade = dfg['Profit_neto'].mean() if trades > 0 else 0
        ret_dd = profit_total / abs(dd_max) if (dd_max not in (0, np.nan, None) and not np.isnan(dd_max)) else np.nan
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
    resumen = pl.DataFrame(resumen_lista)
    return resumen, meses_disponibles

def mostrar_kpis(resumen, dd_max_portafolio, df_portafolio, multiplicador):
    total_profit = (df_portafolio['Profit'] * multiplicador).sum()
    total_commissions = (
        (df_portafolio['Commission'] * multiplicador).sum() +
        (df_portafolio['Taxes'] * multiplicador).sum() +
        (df_portafolio['Swap'] * multiplicador).sum()
    )
    profit_total_neto = total_profit - abs(total_commissions)
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

def descargar_dataframe(df, nombre_archivo, label):
    towrite = io.BytesIO()
    df.to_pandas().to_excel(towrite, index=False, engine='openpyxl')
    towrite.seek(0)
    st.download_button(
        label=label,
        data=towrite,
        file_name=nombre_archivo,
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# ============= CARGA DE ARCHIVOS PRINCIPAL =============

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

# =================== INTERFAZ PRINCIPAL POR TABS ===================

if df is not None and not df.is_empty():
    tab_dashboard, tab_optim = st.tabs(['📊 Dashboard', '🔨 Optimizador de Portafolios'])

    # ====================== TAB DASHBOARD ======================
    with tab_dashboard:
        with pl.StringCache():
            df = df.with_columns([
                pl.col('Open Time').cast(pl.Utf8).str.strptime(pl.Datetime, format="%Y.%m.%d %H:%M:%S", strict=False).alias('Open Time'),
                pl.col('Close Time').cast(pl.Utf8).str.strptime(pl.Datetime, format="%Y.%m.%d %H:%M:%S", strict=False).alias('Close Time'),
                pl.col('Cleaned_EA').cast(pl.Utf8).alias('Cleaned_EA'),
            ])
            df = df.with_columns([
                pl.col('Close Time').dt.strftime('%Y-%m').alias('Mes')
            ])
            excluir_comentarios = ["deposited", "deposit", "withdraw", "withdrawal", "api", "apf", "transfe" ,
                                   "canceled", "cancelled", "cancelado"]
            df = df.filter(
                pl.col('Coment').is_not_null() &
                ~pl.col('Coment').str.to_lowercase().str.contains('|'.join(excluir_comentarios))
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
        multiplicador = st.sidebar.number_input(
            "Multiplicador", value=1.0, min_value=0.0, step=0.1,
        )
        filtro_texto = st.sidebar.text_input("Buscar texto en nombre de EA:", "")
        profit_min = st.sidebar.number_input("Profit total mínimo", value=-99999.0)
        dd_maximo_max = st.sidebar.number_input("DD máximo absoluto límite", value=9999999.0)
        ocultar_no_selec = st.sidebar.checkbox("Ocultar EAs no seleccionados manualmente", value=False)

        mask_fechas = (df['Close Time'].dt.date() >= fecha_desde) & (df['Close Time'].dt.date() <= fecha_hasta)
        df_f = df.filter(mask_fechas)
        if filtro_texto.strip():
            terminos = [t.strip() for t in filtro_texto.replace(",", ";").split(";") if t.strip()]
            if terminos:
                mask = df_f.select(
                    pl.col('Cleaned_EA')
                      .str.strip_chars()
                      .str.to_lowercase()
                      .map_elements(lambda x: any(t.lower() in x for t in terminos), return_dtype=pl.Boolean)
                      .alias('filtro')
                )['filtro']
                df_f = df_f.filter(mask)
        if df_f.is_empty():
            st.warning("No hay datos que coincidan con el filtro actual.")
            st.stop()
        resumen, meses_disponibles = calcular_resumen(df_f, multiplicador)
        resumen = resumen.filter(
            (pl.col('Profit_Total') >= profit_min) &
            (pl.col('DD_Max_Curva') >= -dd_maximo_max)
        )

        st.markdown("---")
        st.markdown("#### DD máximo global (portafolio todas las EAs seleccionadas combinadas, en el periodo)")
        df_portafolio = df_f.filter(pl.col('Cleaned_EA').is_in(resumen['Cleaned_EA']))
        df_portafolio = df_portafolio.sort('Close Time')
        df_portafolio = df_portafolio.with_columns(
            (pl.col('Profit') * multiplicador).alias('Profit_neto'),
            (pl.col('Commission') * multiplicador).alias('Commission_multi'),
            (pl.col('Taxes') * multiplicador).alias('Taxes_multi'),
            (pl.col('Swap') * multiplicador).alias('Swap_multi')
        )
        df_portafolio = df_portafolio.with_columns(
            pl.col('Profit_neto').cum_sum().alias('Saldo_Acumulado')
        )
        saldo = df_portafolio['Saldo_Acumulado'].to_numpy()
        peak = np.maximum.accumulate(saldo)
        dd = saldo - peak
        dd_max_portafolio = dd.min() if dd.size > 0 else np.nan

        st.metric("DD máximo global (portafolio)", f"{dd_max_portafolio:,.2f}")
        mostrar_kpis(resumen, dd_max_portafolio, df_portafolio, multiplicador)

        fig = px.line(df_portafolio.to_pandas(), x='Close Time', y='Saldo_Acumulado', title="Curva de saldo combinada seleccionada")
        st.plotly_chart(fig, use_container_width=True)
        figdd = px.line(df_portafolio.to_pandas(), x='Close Time', y=dd, title="Drawdown global en el tiempo")
        st.plotly_chart(figdd, use_container_width=True)
        st.dataframe(resumen.to_pandas().set_index('Cleaned_EA'), use_container_width=True)

        # =========  DESCARGAS EXCEL ============
        # 1. Descargar resumen
        descargar_dataframe(resumen, "resumen_EAs.xlsx", "Descargar tabla resumen en Excel")

        # 2. Descargar todos los trades en formato estándar
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
            .then(pl.col("Cleaned_EA"))
            .otherwise(pl.col("Coment"))
            .alias("Coment")
        ])
        df_f_export = df_f_export.with_columns([
            pl.col('Open Time').dt.strftime('%Y.%m.%d %H:%M:%S').alias('Open Time'),
            pl.col('Close Time').dt.strftime('%Y.%m.%d %H:%M:%S').alias('Close Time'),
        ])
        descargar_dataframe(df_f_export, "Export_trades_EAs.xlsx", "Descargar Todos Trades en Excel")

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
                            resumen_mes.to_pandas().to_excel(
                                writer, sheet_name=str(mes), index=False)
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
                label="Descarga excel con info mensual resumida",
                data=excel_mensual,
                file_name="resumen_mensual_EAs.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            st.warning("No hay datos suficientes para generar el reporte mensual")

    # ====================== TAB OPTIMIZADOR ======================
    with tab_optim:
        st.header("🔨 Optimizador de Portafolios")
        st.markdown("Busca combinaciones óptimas de EAs según métricas.\n\n**Opciones:** menor DD, mayor profit, más estable, recuperación más rápida")

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
                "Métrica de optimización",
                [
                    "Menor Max DD",
                    "Mayor Profit",
                    "Mayor Profit/DD",
                    "Más Estable (std. mensual)",
                    "Recuperación más Rápida (max DD duration)"
                ],
                index=0
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
                    dfp = dfg.to_pandas()
                    dfp = dfp.sort_values("Close Time")
                    saldo = dfp['Profit'].cumsum().values if not dfp.empty else np.zeros(1)
                    hwm = np.maximum.accumulate(saldo)
                    dd = saldo - hwm
                    max_dd = dd.min() if len(dd) > 0 else 0
                    max_dd_idx = dd.argmin() if len(dd) > 0 else 0
                    max_dd_start = hwm[:max_dd_idx + 1].argmax() if len(hwm) > 0 else 0
                    dd_recovery_idx = None
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
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'


                    )
                # ---------------------------------------------------
                # DESCARGA RESUMEN CLASIFICACIÓN OPTIMIZACIÓN (NUEVO)
                # ---------------------------------------------------
                import xlsxwriter

                if len(resultados_ordenados) > 0:
                    # 1. Preparamos los encabezados:
                    n_eas = portfolio_size
                    # Nombre de columnas para las EAs dinámico según tamaño de portafolio:
                    ea_cols = [f"EA_{i + 1}" for i in range(n_eas)]
                    tabla = []

                    # 2. Construimos la tabla resumen (sólo los top_n ordenados, o todos si quieres)
                    for idx, res in enumerate(resultados_ordenados[:top_n]):
                        fila = [idx + 1]
                        # EAs del portafolio
                        eas = list(res["combo"])
                        # Añado 'None' si son menos que el máximo display
                        eas += [None] * (n_eas - len(eas))
                        fila.extend(eas)
                        # Métricas clave
                        fila.extend([
                            res['max_dd'],
                            res['profit'],
                            res['std_month'],
                            res['max_dd_duration'],
                            res['ret_dd']
                        ])
                        tabla.append(fila)

                    df_out = pd.DataFrame(
                        tabla,
                        columns=(["Rank"] + ea_cols + ["Max DD", "Profit total", "Estabilidad (Std)", "Duración DD",
                                                       "Profit/DD"])
                    )

                    # 3. Montamos el archivo Excel con una primera fila extra de info
                    excel_bytes = io.BytesIO()
                    with pd.ExcelWriter(excel_bytes, engine='xlsxwriter') as writer:
                        info_df = pd.DataFrame([
                            [f"Nº EAs: {n_eas} / Clasificación: {metric_option}"]
                        ])
                        info_df.to_excel(writer, sheet_name="Resumen", header=False, index=False, startrow=0)
                        df_out.to_excel(writer, sheet_name="Resumen", startrow=2, index=False)
                    excel_bytes.seek(0)

                    st.download_button(
                        label="⬇️ Descargar Excel RESUMEN de la Clasificación",
                        data=excel_bytes,
                        file_name="optimizador_resumen.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

else:
    st.info("Por favor, sube archivos de MT4/cTrader y/o StrategyQuant en los botones correspondientes arriba.")
