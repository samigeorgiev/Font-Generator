import React from 'react';

import FGLogo from 'assets/images/FGLogo.png';

import styles from './index.module.css';

const logo = props => (
    <div className={styles.Logo}>
        <img src={FGLogo} alt="FGLogo" />
    </div>
);

export default logo;