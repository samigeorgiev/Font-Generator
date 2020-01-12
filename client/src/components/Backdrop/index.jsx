import React from 'react';

import styles from './index.module.css';

const backdrop = props => (
    <div className={styles.Backdrop} onClick={props.click} />
);

export default backdrop;